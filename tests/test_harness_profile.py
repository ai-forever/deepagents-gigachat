"""Tests for the GigaChat harness profile package."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from deepagents.backends import LocalShellBackend
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.profiles.harness.harness_profiles import _get_harness_profile
from gigachat.exceptions import UnprocessableEntityError
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents_gigachat import (
    GIGACHAT_CONTEXT_WINDOWS,
    ContextWindowGuardMiddleware,
    DeterministicOutputMiddleware,
    LoopBreakerMiddleware,
    ShellSafetyMiddleware,
    SpecificationAuditMiddleware,
    ToolContractMiddleware,
    build_system_prompt,
    get_initial_workspace_files,
    get_workspace_path,
    harness_profile,
    register_harness,
    set_workspace_path,
)
from deepagents_gigachat.harness_profile import MemoryTaskMiddleware


@pytest.fixture(autouse=True)
def _reset_workspace_context() -> Any:
    set_workspace_path(None)
    yield
    set_workspace_path(None)


def test_public_api_exports_register_harness() -> None:
    assert register_harness is harness_profile.register_harness
    assert build_system_prompt("external_runtime")


def test_entry_point_is_declared() -> None:
    eps = entry_points(group="deepagents.harness_profiles")

    assert any(
        ep.name == "gigachat" and ep.value == "deepagents_gigachat:register_harness"
        for ep in eps
    )


def test_register_harness_registers_gigachat_profile() -> None:
    harness_profile.register_harness()

    profile = _get_harness_profile("gigachat:GigaChat-3-Ultra")

    assert profile is not None
    assert "write_file" in profile.tool_description_overrides
    assert "relative path" in profile.tool_description_overrides["write_file"]
    assert "foo.py" in profile.tool_description_overrides["write_file"]
    assert "src/foo.py" in profile.tool_description_overrides["write_file"]


def test_register_harness_uses_both_provider_aliases(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    assert {"gigachat", "giga"} == set(captured)
    profile = captured["gigachat"]
    assert captured["giga"] is profile
    assert "relative to the workspace root" in profile.base_system_prompt
    assert "AGENTS.md" in profile.base_system_prompt
    assert "read_file MEMORY.md" in profile.base_system_prompt
    assert "single-quoted heredoc" in profile.base_system_prompt
    assert "relative path" in profile.tool_description_overrides["write_file"]
    assert "RELATIVE paths" in profile.tool_description_overrides["execute"]
    middleware_names = {
        type(middleware).__name__
        for middleware in profile.materialize_extra_middleware()
    }
    assert "PathNormalizerMiddleware" in middleware_names
    assert "ContextWindowGuardMiddleware" in middleware_names
    assert "DeterministicOutputMiddleware" in middleware_names
    assert "SpecificationAuditMiddleware" in middleware_names
    assert "MemoryTaskMiddleware" in middleware_names
    assert "AgentsMdInjectMiddleware" not in middleware_names


def test_register_harness_configures_context_guard(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        lambda provider, profile: captured.setdefault(provider, profile),
    )

    harness_profile.register_harness(
        context_window=64_000,
        summarization_trigger=0.75,
    )

    guard = next(
        middleware
        for middleware in captured["gigachat"].materialize_extra_middleware()
        if isinstance(middleware, ContextWindowGuardMiddleware)
    )
    assert guard.context_window == 64_000
    assert guard.trigger_fraction == 0.75
    assert guard.trigger_tokens == 48_000


def test_register_harness_can_use_external_runtime_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness(
        profile_variant="external_runtime",
        tool_contract="Visible tools: runtime_read, runtime_write, runtime_answer.",
    )

    profile = captured["gigachat"]
    assert "External runtime tools" in profile.base_system_prompt
    assert "read_file once" not in profile.base_system_prompt
    assert "write_file" not in profile.tool_description_overrides
    assert any(
        isinstance(m, ToolContractMiddleware)
        for m in profile.materialize_extra_middleware()
    )


def test_shell_safety_middleware_supports_async_tool_calls() -> None:
    middleware = ShellSafetyMiddleware()
    request = type(
        "Request",
        (),
        {
            "tool_call": {
                "id": "call_1",
                "name": "execute",
                "args": {"command": 'python -c "x=0; for v in [1]: x += v"'},
            }
        },
    )()

    async def handler(_request: Any) -> ToolMessage:
        raise AssertionError("unsafe shell command should be blocked before handler")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert isinstance(result, ToolMessage)
    assert result.name == "execute"
    assert result.tool_call_id == "call_1"
    assert "[SHELL-SAFETY]" in result.content


def test_context_guard_requests_compaction_for_unprofiled_model() -> None:
    middleware = ContextWindowGuardMiddleware(
        context_window=400,
        trigger_fraction=0.5,
    )
    request = SimpleNamespace(
        model=SimpleNamespace(profile=None),
        messages=[HumanMessage(content="x" * 200) for _ in range(7)],
        system_message=SystemMessage(content="system"),
        tools=[],
    )

    with pytest.raises(ContextOverflowError, match="proactive compaction"):
        middleware.wrap_model_call(request, lambda _request: "unreachable")


def test_context_guard_defers_to_stock_summarizer_for_profiled_model() -> None:
    middleware = ContextWindowGuardMiddleware(
        context_window=400,
        trigger_fraction=0.5,
    )
    request = SimpleNamespace(
        model=SimpleNamespace(profile={"max_input_tokens": 400}),
        messages=[HumanMessage(content="x" * 200) for _ in range(7)],
        system_message=None,
        tools=[],
    )

    assert middleware.wrap_model_call(request, lambda _request: "ok") == "ok"


def test_context_guard_does_not_assume_an_unprofiled_context_window() -> None:
    middleware = ContextWindowGuardMiddleware()
    request = SimpleNamespace(
        model=SimpleNamespace(profile=None, model="Unknown-GigaChat-Model"),
        messages=[HumanMessage(content="x" * 200_000) for _ in range(7)],
        system_message=None,
        tools=[],
    )

    assert middleware.wrap_model_call(request, lambda _request: "ok") == "ok"
    assert middleware.context_window is None
    assert middleware.trigger_tokens is None


@pytest.mark.parametrize(
    "model_name",
    [
        "GigaChat",
        "GigaChat-Pro",
        "GigaChat-Max",
        "GigaChat-2",
        "GigaChat-2-Pro",
        "GigaChat-2-Max",
        "GigaChat-2-Reasoning",
        "GigaChat-3-Lightning",
        "GigaChat-3-Pro",
        "GigaChat-3-Ultra",
        "GigaChat-2-Pro-preview",
        "GigaChat-3-Ultra:32.3.18.5",
        "gigachat:GigaChat-2-Max",
    ],
)
def test_context_guard_knows_gigachat_generation_models(model_name: str) -> None:
    middleware = ContextWindowGuardMiddleware()
    model = SimpleNamespace(profile=None, model=model_name)

    assert middleware.context_window_for_model(model) == 261_120


def test_public_context_window_table_covers_current_model_families() -> None:
    assert GIGACHAT_CONTEXT_WINDOWS["GigaChat-2"] == 261_120
    assert GIGACHAT_CONTEXT_WINDOWS["GigaChat-2-Reasoning"] == 261_120
    assert GIGACHAT_CONTEXT_WINDOWS["GigaChat-3-Lightning"] == 261_120
    assert GIGACHAT_CONTEXT_WINDOWS["GigaChat-3-Pro"] == 261_120
    assert GIGACHAT_CONTEXT_WINDOWS["GigaChat-3-Ultra"] == 261_120


def test_context_guard_translates_gigachat_payload_overflow() -> None:
    middleware = ContextWindowGuardMiddleware()
    request = SimpleNamespace(
        model=SimpleNamespace(profile=None),
        messages=[],
        system_message=None,
        tools=[],
    )
    provider_error = UnprocessableEntityError(
        url="https://gigachat.sberdevices.ru/v1/chat/completions",
        status_code=422,
        content=(
            b'{"status":422,"message":"CONTEXT_TOO_LONG: context too long '
            b'301216, maximum allowed context is 261120"}\n'
        ),
        headers=httpx.Headers({"content-type": "application/json"}),
    )

    def raise_provider_error(_request: Any) -> None:
        raise provider_error

    with pytest.raises(ContextOverflowError, match="context window") as raised:
        middleware.wrap_model_call(request, raise_provider_error)

    assert raised.value.__cause__ is provider_error


def test_context_guard_drives_stock_summarization_retry(tmp_path: Path) -> None:
    model = FakeListChatModel(responses=["summary"])
    summarization = create_summarization_middleware(
        model,
        LocalShellBackend(root_dir=tmp_path, virtual_mode=True),
    )
    guard = ContextWindowGuardMiddleware(
        context_window=400,
        trigger_fraction=0.5,
    )
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="x" * 200) for _ in range(7)],
        state={},
    )
    model_requests: list[Any] = []

    def model_handler(compacted_request: Any) -> ModelResponse:
        model_requests.append(compacted_request)
        return ModelResponse(result=[AIMessage(content="final")])

    result = summarization.wrap_model_call(
        request,
        lambda compacted_request: guard.wrap_model_call(
            compacted_request,
            model_handler,
        ),
    )

    assert len(model_requests) == 1
    assert model_requests[0].messages[0].additional_kwargs["lc_source"] == "summarization"
    assert result.command is not None
    assert "_summarization_event" in result.command.update


def test_workspace_path_is_isolated_by_execution_context(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "first.txt").write_text("one", encoding="utf-8")
    (second / "second.txt").write_text("two", encoding="utf-8")
    set_workspace_path(first)

    other_context = contextvars.copy_context()
    other_context.run(set_workspace_path, second)

    assert get_workspace_path() == first
    assert other_context.run(get_workspace_path) == second
    assert get_initial_workspace_files() == frozenset({"first.txt"})
    assert other_context.run(get_initial_workspace_files) == frozenset({"second.txt"})


def test_workspace_middleware_remains_bound_in_fresh_thread(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        lambda provider, profile: captured.setdefault(provider, profile),
    )

    # Match the benchmark runner's order: register first, select a workspace,
    # then build/materialize the agent graph before invoking it in a new thread.
    harness_profile.register_harness()
    (tmp_path / "AGENTS.md").write_text("memory instructions", encoding="utf-8")
    (tmp_path / "input.csv").write_text("value\n1\n", encoding="utf-8")
    set_workspace_path(tmp_path)
    middleware = captured["gigachat"].materialize_extra_middleware()
    memory = next(item for item in middleware if isinstance(item, MemoryTaskMiddleware))
    audit = next(
        item for item in middleware if isinstance(item, SpecificationAuditMiddleware)
    )
    deterministic = next(
        item for item in middleware if isinstance(item, DeterministicOutputMiddleware)
    )
    results: dict[str, Any] = {}

    def invoke_in_fresh_thread() -> None:
        results["ambient_workspace"] = get_workspace_path()
        results["memory"] = memory.before_model(
            {"messages": [HumanMessage(content="Use my saved preferences.")]},
            runtime=None,  # type: ignore[arg-type]
        )
        input_state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"file_path": "input.csv"},
                            "id": "read_input",
                        }
                    ],
                ),
                ToolMessage(
                    content="value\n1\n",
                    tool_call_id="read_input",
                    name="read_file",
                ),
            ]
        }
        results["input_audit"] = audit.before_model(
            input_state,
            runtime=None,  # type: ignore[arg-type]
        )
        request = SimpleNamespace(
            tool_call={
                "id": "write_total",
                "name": "write_file",
                "args": {"file_path": "total.txt", "content": "7"},
            },
            state={"messages": []},
        )
        results["write"] = deterministic.wrap_tool_call(
            request,
            lambda _request: (_ for _ in ()).throw(
                AssertionError("derived scalar write must be blocked")
            ),
        )

    thread = threading.Thread(target=invoke_in_fresh_thread, name="hb-agent-invoke")
    thread.start()
    thread.join()

    assert results["ambient_workspace"] is None
    assert "[MEMORY-TASK]" in results["memory"]["messages"][0].content
    assert results["input_audit"] is None
    assert "[DETERMINISTIC-OUTPUT]" in results["write"].content


def test_workspace_snapshot_is_bounded_and_skips_vendor_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "index").write_text("metadata", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dependency.js").write_text(
        "vendor", encoding="utf-8"
    )

    set_workspace_path(tmp_path)

    assert get_initial_workspace_files() == frozenset({"input.txt"})


def test_workspace_dependent_nudges_are_disabled_without_workspace() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "input.json"},
                        "id": "read_input",
                    }
                ],
            ),
            ToolMessage(content="{}", tool_call_id="read_input", name="read_file"),
        ]
    }

    assert DeterministicOutputMiddleware().before_model(
        state, runtime=None  # type: ignore[arg-type]
    ) is None
    assert SpecificationAuditMiddleware().before_model(
        state, runtime=None  # type: ignore[arg-type]
    ) is None


def test_deterministic_output_guard_blocks_mental_scalar_write(tmp_path: Path) -> None:
    (tmp_path / "input.csv").write_text("value\n1\n", encoding="utf-8")
    set_workspace_path(tmp_path)
    middleware = DeterministicOutputMiddleware()
    request = type(
        "Request",
        (),
        {
            "tool_call": {
                "id": "call_1",
                "name": "write_file",
                "args": {"file_path": "total.txt", "content": "7"},
            },
            "state": {
                "messages": [
                    HumanMessage(content="Count values."),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "input.csv"},
                                "id": "read_1",
                            }
                        ],
                    ),
                ]
            },
        },
    )()

    result = middleware.wrap_tool_call(
        request,
        lambda _request: (_ for _ in ()).throw(AssertionError("must be blocked")),
    )

    assert "[DETERMINISTIC-OUTPUT]" in result.content


def test_deterministic_output_guard_allows_write_after_python_execute(tmp_path: Path) -> None:
    (tmp_path / "input.csv").write_text("value\n1\n", encoding="utf-8")
    set_workspace_path(tmp_path)
    middleware = DeterministicOutputMiddleware()
    expected = ToolMessage(content="ok", tool_call_id="call_2", name="write_file")
    request = type(
        "Request",
        (),
        {
            "tool_call": {
                "id": "call_2",
                "name": "write_file",
                "args": {"file_path": "total.txt", "content": "1"},
            },
            "state": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "execute",
                                "args": {"command": "python3 <<'PY'\nprint(1)\nPY"},
                                "id": "run_1",
                            }
                        ],
                    )
                ]
            },
        },
    )()

    assert middleware.wrap_tool_call(request, lambda _request: expected) is expected


def test_deterministic_next_step_nudge_follows_observation(tmp_path: Path) -> None:
    set_workspace_path(tmp_path)
    middleware = DeterministicOutputMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Derive an output from input.txt."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "input.txt"},
                        "id": "read_1",
                    }
                ],
            ),
            ToolMessage(content="data", tool_call_id="read_1", name="read_file"),
        ]
    }

    result = middleware.before_model(state, runtime=None)  # type: ignore[arg-type]

    assert result is not None
    assert "NEXT tool must be `execute`" in result["messages"][0].content


def test_deterministic_nudge_is_suppressed_for_skill_workspace(tmp_path: Path) -> None:
    (tmp_path / ".agents" / "skills").mkdir(parents=True)
    set_workspace_path(tmp_path)
    middleware = DeterministicOutputMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Apply the relevant skill."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": ".agents/skills/example/SKILL.md"},
                        "id": "read_skill",
                    }
                ],
            ),
            ToolMessage(content="rules", tool_call_id="read_skill", name="read_file"),
        ]
    }

    assert middleware.before_model(state, runtime=None) is None  # type: ignore[arg-type]


def test_specification_audit_targets_only_newly_created_outputs(tmp_path: Path) -> None:
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    set_workspace_path(tmp_path)
    middleware = SpecificationAuditMiddleware()
    input_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "input.json"},
                        "id": "read_input",
                    }
                ],
            ),
            ToolMessage(content="{}", tool_call_id="read_input", name="read_file"),
        ]
    }
    assert middleware.before_model(input_state, runtime=None) is None  # type: ignore[arg-type]

    output_state = {
        "messages": [
            HumanMessage(content="Create report.json with exact fields."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "report.json"},
                        "id": "read_output",
                    }
                ],
            ),
            ToolMessage(content='{"value": 1}', tool_call_id="read_output", name="read_file"),
        ]
    }
    result = middleware.before_model(output_state, runtime=None)  # type: ignore[arg-type]
    assert result is not None
    assert "Exact value types" in result["messages"][0].content


def test_loop_breaker_nudge_markers_are_detected_across_ai_turns() -> None:
    messages = [
        HumanMessage(content="[BUDGET-NUDGE-BATCH] switch strategy"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
        ToolMessage(content="ok", tool_call_id="1", name="read_file"),
    ]

    assert LoopBreakerMiddleware._already_nudged(messages, "[BUDGET-NUDGE-BATCH]")


def test_loop_breaker_uses_separate_bounded_markers() -> None:
    middleware = LoopBreakerMiddleware()
    messages: list[Any] = [HumanMessage(content="[LOOP-BREAKER-GREP] switch search")]
    for index in range(3):
        call_id = f"execute_{index}"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute",
                            "args": {"command": "broken command"},
                            "id": call_id,
                        }
                    ],
                ),
                ToolMessage(
                    content="SyntaxError: invalid syntax",
                    tool_call_id=call_id,
                    name="execute",
                ),
            ]
        )

    result = middleware.before_model(
        {"messages": messages}, runtime=None  # type: ignore[arg-type]
    )

    assert result is not None
    assert "[LOOP-BREAKER-ERROR]" in result["messages"][0].content
    capped_messages = [
        HumanMessage(content="[LOOP-BREAKER-ERROR] first"),
        HumanMessage(content="[LOOP-BREAKER-ERROR] second"),
        *messages[1:],
    ]
    assert middleware.before_model(
        {"messages": capped_messages}, runtime=None  # type: ignore[arg-type]
    ) is None
