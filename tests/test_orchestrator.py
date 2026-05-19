"""Tests for generic routed orchestration helpers."""

from __future__ import annotations

import os

from pytest import MonkeyPatch

from deepagents_gigachat.orchestrator import build_deep_agent_env, run_routed, temporary_env
from deepagents_gigachat.routing import RoutingDecision


def test_run_routed_dispatches_direct_without_touching_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPAGENTS_GIGACHAT_PROFILE", raising=False)
    calls: list[str] = []

    def direct_runner(task: str, **_kwargs: object) -> str:
        calls.append(f"direct:{task}")
        return "direct"

    def deep_runner(task: str, **_kwargs: object) -> str:
        calls.append(f"deep:{task}")
        return "deep"

    result = run_routed(
        "task-x",
        decision=RoutingDecision(execution_route="direct", tool_route="data"),
        direct_runner=direct_runner,
        deep_runner=deep_runner,
        direct_kwargs={"alpha": 1},
        deep_env={"DEEPAGENTS_GIGACHAT_PROFILE": "hybrid"},
    )

    assert result == "direct"
    assert calls == ["direct:task-x"]
    assert os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE") is None


def test_run_routed_dispatches_deep_with_temporary_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPAGENTS_GIGACHAT_PROFILE", raising=False)
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)
    seen: list[tuple[str | None, str | None]] = []

    def direct_runner(task: str, **_kwargs: object) -> str:
        raise AssertionError(f"unexpected direct route for {task}")

    def deep_runner(task: str, **_kwargs: object) -> str:
        seen.append(
            (
                os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE"),
                os.environ.get("GIGACHAT_MODEL"),
            )
        )
        return f"deep:{task}"

    result = run_routed(
        "task-y",
        decision=RoutingDecision(execution_route="deep", tool_route="hybrid"),
        direct_runner=direct_runner,
        deep_runner=deep_runner,
        deep_env=build_deep_agent_env(
            model_name="GigaChat-Test",
            deep_profile="adaptive-tools",
        ),
    )

    assert result == "deep:task-y"
    assert seen == [("adaptive-tools", "GigaChat-Test")]
    assert os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE") is None
    assert os.environ.get("GIGACHAT_MODEL") is None


def test_temporary_env_restores_existing_values(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_GIGACHAT_PROFILE", "pi-lite")

    with temporary_env({"DEEPAGENTS_GIGACHAT_PROFILE": "hybrid"}):
        assert os.environ["DEEPAGENTS_GIGACHAT_PROFILE"] == "hybrid"

    assert os.environ["DEEPAGENTS_GIGACHAT_PROFILE"] == "pi-lite"
