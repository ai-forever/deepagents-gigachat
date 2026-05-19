"""Tests for retry handling in the direct GigaChat runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_bench import runner
from harness_bench.core import Task, VerifyResult


def test_is_transient_agent_error_detects_transport_failures() -> None:
    assert runner._is_transient_agent_error(  # noqa: SLF001
        "httpx.RemoteProtocolError: Server disconnected without sending a response."
    )
    assert runner._is_transient_agent_error(  # noqa: SLF001
        "gigachat.exceptions.AuthenticationError: 401 https://example.test"
    )
    assert runner._is_transient_agent_error(  # noqa: SLF001
        "During task with name 'model' and id 'abc'"
    )
    assert not runner._is_transient_agent_error(  # noqa: SLF001
        "GRAPH_RECURSION_LIMIT"
    )
    assert runner._is_graph_recursion_error("GRAPH_RECURSION_LIMIT")  # noqa: SLF001


def test_run_task_retries_transient_agent_error(monkeypatch: Any) -> None:
    attempts = 0

    class FakeAgent:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace

        def invoke(self, _payload: dict[str, Any]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError(
                    "httpx.RemoteProtocolError: "
                    "Server disconnected without sending a response."
                )
            (self.workspace / "ok.txt").write_text("ok\n")

    def fake_build_agent(workspace: Path, *, recursion_limit: int = 80) -> FakeAgent:
        return FakeAgent(workspace)

    task = Task(
        id="task_retry",
        name="retry",
        prompt="write ok",
        verifier=lambda ws: VerifyResult(
            (ws / "ok.txt").read_text() == "ok\n",
            "ok",
        ),
    )

    monkeypatch.setattr(runner, "build_agent", fake_build_agent)

    result = runner.run_task(
        task,
        agent_error_retries=1,
        retry_base_delay=0,
    )

    assert result.passed
    assert attempts == 2


def test_run_task_uses_correction_retry_after_verifier_failure(
    monkeypatch: Any,
) -> None:
    prompts: list[str] = []

    class FakeAgent:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace

        def invoke(self, payload: dict[str, Any]) -> None:
            prompt = payload["messages"][0]["content"]
            prompts.append(prompt)
            if len(prompts) == 1:
                (self.workspace / "ok.txt").write_text("bad\n")
                return
            (self.workspace / "ok.txt").write_text("ok\n")

    def fake_build_agent(workspace: Path, *, recursion_limit: int = 80) -> FakeAgent:
        return FakeAgent(workspace)

    def verify(workspace: Path) -> VerifyResult:
        output = workspace / "ok.txt"
        if not output.exists():
            return VerifyResult(False, "ok.txt missing")
        if output.read_text() != "ok\n":
            return VerifyResult(False, "ok.txt content differs")
        return VerifyResult(True, "ok")

    task = Task(
        id="task_correction",
        name="correction",
        prompt="write ok",
        verifier=verify,
    )

    monkeypatch.setattr(runner, "build_agent", fake_build_agent)

    result = runner.run_task(
        task,
        correction_retries=1,
    )

    assert result.passed
    assert len(prompts) == 2
    assert "write ok" in prompts[1]
    assert "ok.txt content differs" in prompts[1]


def test_run_task_uses_finalization_after_verifier_failure(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[int, str]] = []

    class FakeAgent:
        def __init__(self, workspace: Path, recursion_limit: int) -> None:
            self.workspace = workspace
            self.recursion_limit = recursion_limit

        def invoke(self, payload: dict[str, Any]) -> None:
            prompt = payload["messages"][0]["content"]
            calls.append((self.recursion_limit, prompt))
            if len(calls) == 1:
                (self.workspace / "ok.txt").write_text("bad\n")
                return
            (self.workspace / "ok.txt").write_text("ok\n")

    def fake_build_agent(
        workspace: Path,
        *,
        recursion_limit: int = 80,
    ) -> FakeAgent:
        return FakeAgent(workspace, recursion_limit)

    def verify(workspace: Path) -> VerifyResult:
        output = workspace / "ok.txt"
        if not output.exists():
            return VerifyResult(False, "ok.txt missing")
        if output.read_text() != "ok\n":
            return VerifyResult(False, "ok.txt content differs")
        return VerifyResult(True, "ok")

    task = Task(
        id="task_finalization",
        name="finalization",
        prompt="write ok",
        verifier=verify,
    )

    monkeypatch.setattr(runner, "build_agent", fake_build_agent)

    result = runner.run_task(
        task,
        finalization_retries=1,
    )

    assert result.passed
    assert [limit for limit, _ in calls] == [80, 20]
    assert "Finalization pass" in calls[1][1]
    assert "write ok" in calls[1][1]
    assert "ok.txt content differs" in calls[1][1]


def test_run_task_recovers_after_graph_recursion_limit(monkeypatch: Any) -> None:
    calls: list[tuple[int, str]] = []

    class FakeAgent:
        def __init__(self, workspace: Path, recursion_limit: int) -> None:
            self.workspace = workspace
            self.recursion_limit = recursion_limit

        def invoke(self, payload: dict[str, Any]) -> None:
            prompt = payload["messages"][0]["content"]
            calls.append((self.recursion_limit, prompt))
            if len(calls) == 1:
                (self.workspace / "ok.txt").write_text("bad\n")
                raise RuntimeError("GRAPH_RECURSION_LIMIT")
            (self.workspace / "ok.txt").write_text("ok\n")

    def fake_build_agent(
        workspace: Path,
        *,
        recursion_limit: int = 80,
    ) -> FakeAgent:
        return FakeAgent(workspace, recursion_limit)

    def verify(workspace: Path) -> VerifyResult:
        output = workspace / "ok.txt"
        if not output.exists():
            return VerifyResult(False, "ok.txt missing")
        if output.read_text() != "ok\n":
            return VerifyResult(False, "ok.txt content differs")
        return VerifyResult(True, "ok")

    task = Task(
        id="task_recursion_recovery",
        name="recursion recovery",
        prompt="write ok",
        verifier=verify,
    )

    monkeypatch.setattr(runner, "build_agent", fake_build_agent)

    result = runner.run_task(
        task,
        recursion_limit=80,
        recursion_recovery_attempts=1,
    )

    assert result.passed
    assert [limit for limit, _ in calls] == [80, 20]
    assert "write ok" in calls[1][1]
    assert "ok.txt content differs" in calls[1][1]
