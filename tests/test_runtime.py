"""Tests for the public routed workspace runtime."""

from __future__ import annotations

import os
from pathlib import Path

from pytest import MonkeyPatch

from deepagents_gigachat import runtime


class _FakeModel:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[list[dict[str, str]]] = []

    def invoke(self, payload: list[dict[str, str]]) -> str:
        self.calls.append(payload)
        return self.response_text


class _FakeAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return {"messages": [{"role": "assistant", "content": "done"}]}


def test_invoke_direct_writes_required_artifact(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    model = _FakeModel(
        '{"kind":"python","code":"from pathlib import Path\\nPath(\\"total.txt\\").write_text(\\"15\\\\n\\", encoding=\\"utf-8\\")"}'
    )
    monkeypatch.setattr(runtime, "build_model", lambda **_kwargs: model)

    result = runtime.invoke_direct(
        "Read numbers.csv and write the sum to total.txt.",
        workspace=tmp_path,
        ensure_auth=False,
    )

    assert result.decision.execution_route == "direct"
    assert result.controller_action == {
        "kind": "python",
        "code": 'from pathlib import Path\nPath("total.txt").write_text("15\\n", encoding="utf-8")',
    }
    assert result.action_result is not None
    assert result.action_result.returncode == 0
    assert (tmp_path / "total.txt").read_text(encoding="utf-8") == "15\n"
    assert model.calls


def test_invoke_routed_dispatches_deep_branch(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    agent = _FakeAgent()
    seen: dict[str, str | None] = {}

    def fake_build_deep_agent(workspace: str | Path, **_kwargs: object) -> _FakeAgent:
        seen["workspace"] = str(Path(workspace).resolve())
        seen["profile"] = os.environ.get("DEEPAGENTS_GIGACHAT_PROFILE")
        seen["model"] = os.environ.get("GIGACHAT_MODEL")
        return agent

    monkeypatch.setattr(runtime, "build_deep_agent", fake_build_deep_agent)

    result = runtime.invoke_routed(
        "Implement Stack so pytest passes.",
        workspace=tmp_path,
        model_name="GigaChat-Test",
        deep_profile="adaptive-tools",
        ensure_auth=False,
        load_env=False,
    )

    assert result.decision.execution_route == "deep"
    assert result.decision.tool_route == "hybrid"
    assert result.deep_state == {"messages": [{"role": "assistant", "content": "done"}]}
    assert seen == {
        "workspace": str(tmp_path.resolve()),
        "profile": "adaptive-tools",
        "model": "GigaChat-Test",
    }
    assert agent.calls == [
        {"messages": [{"role": "user", "content": "Implement Stack so pytest passes."}]}
    ]


def test_load_env_from_dotenv_prefers_workspace_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPAGENTS_GIGACHAT_RUNTIME_TEST=loaded\n", encoding="utf-8")
    monkeypatch.delenv("DEEPAGENTS_GIGACHAT_RUNTIME_TEST", raising=False)

    runtime.load_env_from_dotenv(tmp_path)

    assert os.environ.get("DEEPAGENTS_GIGACHAT_RUNTIME_TEST") == "loaded"
