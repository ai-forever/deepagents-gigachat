"""Tests for the semantic benchmark router."""

from __future__ import annotations

import os
from typing import Any

from harness_bench import runner_router
from harness_bench.core import Task, VerifyResult
from harness_bench.runner import TaskRun


def _task(*, task_id: str = "task_x", prompt: str = "do it", tags: tuple[str, ...]) -> Task:
    return Task(
        id=task_id,
        name=task_id,
        prompt=prompt,
        tags=tags,
        verifier=lambda _ws: VerifyResult(True, "ok"),
    )


def test_route_for_task_sends_code_semantics_to_deep() -> None:
    assert runner_router.route_for_task(_task(tags=("python", "impl"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("python", "pytest"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("python", "fix"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("python", "refactor"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("toml", "edit"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("python", "create"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("python", "edit"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("logs", "filter"))) == "deep"
    assert runner_router.route_for_task(_task(tags=("xlsx", "csv", "json"))) == "deep"
    assert (
        runner_router.route_for_task(
            _task(prompt="Bump version in pyproject.toml", tags=("config", "edit"))
        )
        == "deep"
    )


def test_route_for_task_sends_data_and_search_to_direct() -> None:
    assert runner_router.route_for_task(_task(tags=("csv", "compute"))) == "direct"
    assert runner_router.route_for_task(_task(tags=("grep", "search"))) == "direct"
    assert runner_router.route_for_task(_task(tags=("filesystem", "edit"))) == "direct"
    assert (
        runner_router.route_for_task(_task(tags=("filesystem", "edit", "refactor")))
        == "direct"
    )
    assert (
        runner_router.route_for_task(
            _task(prompt="Перенеси функцию helper() из a.py в b.py", tags=("python", "refactor"))
        )
        == "direct"
    )
    assert (
        runner_router.route_for_task(_task(tags=("python", "execute", "compute")))
        == "direct"
    )


def test_route_for_task_can_ignore_benchmark_hints() -> None:
    assert (
        runner_router.route_for_task(
            _task(
                prompt="Count .py files into count.txt",
                tags=("python", "impl"),
            ),
            use_routing_hints=False,
        )
        == "direct"
    )
    assert (
        runner_router.route_for_task(
            _task(
                prompt="Implement src/utils.py:slugify",
                tags=("csv", "compute"),
            ),
            use_routing_hints=False,
        )
        == "deep"
    )


def test_route_for_task_can_use_model_router(monkeypatch: Any) -> None:
    class _FakeRouterModel:
        def invoke(self, _payload: list[dict[str, str]]) -> str:
            return '{"execution_route":"deep","tool_route":"hybrid"}'

    monkeypatch.setattr(runner_router, "_build_router_model", lambda **_kwargs: _FakeRouterModel())

    assert (
        runner_router.route_for_task(
            _task(prompt="Please route this ambiguous task.", tags=()),
            router_mode="model",
            router_model_name="GigaChat-Router",
        )
        == "deep"
    )


def test_run_task_router_dispatches_direct(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_direct(task: Task, **_kwargs: Any) -> TaskRun:
        calls.append(task.id)
        return TaskRun(task.id, True, "direct", 0.1)

    monkeypatch.setattr(runner_router, "run_task_direct", fake_direct)

    result = runner_router.run_task_router(_task(tags=("csv", "compute")))

    assert result.passed
    assert result.message == "direct"
    assert calls == ["task_x"]


def test_run_task_router_dispatches_deep_with_profile_env(monkeypatch: Any) -> None:
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_deep(task: Task, **_kwargs: Any) -> TaskRun:
        calls.append(
            (
                task.id,
                os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE"),
                os.environ.get("GIGACHAT_MODEL"),
            )
        )
        return TaskRun(task.id, True, "deep", 0.1)

    monkeypatch.delenv("DEEPAGENTS_GIGACHAT_PROFILE", raising=False)
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)
    monkeypatch.setattr(runner_router, "run_task", fake_deep)

    result = runner_router.run_task_router(
        _task(tags=("python", "impl")),
        model_name="GigaChat-Test",
        deep_profile="hybrid",
    )

    assert result.passed
    assert calls == [("task_x", "hybrid", "GigaChat-Test")]
    assert os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE") is None
    assert os.environ.get("GIGACHAT_MODEL") is None


def test_run_all_router_sets_shared_deep_env_without_per_task_override(
    monkeypatch: Any,
) -> None:
    direct = _task(task_id="task_01_direct", tags=("csv", "compute"))
    deep = _task(task_id="task_02_deep", tags=("python", "impl"))
    seen: list[tuple[str, str | None, str | None, str | None]] = []

    def fake_run_task_router(task: Task, **kwargs: Any) -> TaskRun:
        seen.append(
            (
                task.id,
                kwargs["deep_profile"],
                os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE"),
                os.environ.get("GIGACHAT_MODEL"),
            )
        )
        return TaskRun(task.id, True, "ok", 0.1)

    monkeypatch.setattr(runner_router, "ALL_TASKS", [direct, deep])
    monkeypatch.setattr(runner_router, "_load_env_from_dotenv", lambda: None)
    monkeypatch.setattr(runner_router, "_ensure_credentials", lambda: None)
    monkeypatch.setattr(runner_router, "run_task_router", fake_run_task_router)
    monkeypatch.delenv("DEEPAGENTS_GIGACHAT_PROFILE", raising=False)
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)

    results = runner_router.run_all_router(
        model_name="GigaChat-Test",
        deep_profile="hybrid",
    )

    assert [result.task_id for result in results] == ["task_01_direct", "task_02_deep"]
    assert seen == [
        ("task_01_direct", None, "hybrid", "GigaChat-Test"),
        ("task_02_deep", None, "hybrid", "GigaChat-Test"),
    ]
    assert os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE") is None
    assert os.environ.get("GIGACHAT_MODEL") is None


def test_temporary_env_restores_existing_values(monkeypatch: Any) -> None:
    monkeypatch.setenv("DEEPAGENTS_GIGACHAT_PROFILE", "pi-lite")

    with runner_router._temporary_env({"DEEPAGENTS_GIGACHAT_PROFILE": "hybrid"}):  # noqa: SLF001
        assert os.environ["DEEPAGENTS_GIGACHAT_PROFILE"] == "hybrid"

    assert os.environ["DEEPAGENTS_GIGACHAT_PROFILE"] == "pi-lite"
