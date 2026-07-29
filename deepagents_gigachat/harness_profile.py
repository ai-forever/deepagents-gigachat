"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from deepagents import (
    HarnessProfile,
    register_harness_profile,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from pydantic import Field

from deepagents_gigachat.prompts import build_system_prompt

GIGACHAT_CONTEXT_WINDOWS: dict[str, int] = {
    # Current public API aliases for the GigaChat 2 family.
    "GigaChat": 128_000,
    "GigaChat-Lite": 128_000,
    "GigaChat-Pro": 128_000,
    "GigaChat-Max": 128_000,
    "GigaChat-2": 128_000,
    "GigaChat-2-Lite": 128_000,
    "GigaChat-2-Pro": 128_000,
    "GigaChat-2-Max": 128_000,
    # Current public GigaChat 3 model.
    "GigaChat-3-Ultra": 128_000,
}
_NORMALIZED_CONTEXT_WINDOWS = {
    model_name.casefold(): context_window
    for model_name, context_window in GIGACHAT_CONTEXT_WINDOWS.items()
}


@tool("think")
def _think(thought: str = Field(..., description="A thought to think about.")) -> str:
    """Use this tool as scratchpad to structure intermediate reasoning."""
    return thought


class ThinkToolMiddleware(AgentMiddleware):
    """Inject the local `think` tool into the default toolset."""

    tools = [_think]


class ToolContractMiddleware(AgentMiddleware):
    """Append an explicit runtime tool contract near the user messages.

    DeepAgents core middleware may add generic filesystem instructions before a
    profile-level tool exclusion runs. This middleware gives harnesses a
    provider-agnostic escape hatch: state the tools that are actually usable in
    this run, without patching DeepAgents itself.
    """

    name = "ToolContractMiddleware"

    def __init__(self, contract: str | None = None) -> None:
        self.contract = (contract or "").strip()

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        if not self.contract:
            return None
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None
        marker = "[TOOL-CONTRACT]"
        for msg in messages:
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and marker in content:
                return None
        return {"messages": [HumanMessage(content=f"{marker}\n{self.contract}")]}


class ShellSafetyMiddleware(AgentMiddleware):
    """Block common unsafe shell serialization patterns before execution."""

    name = "ShellSafetyMiddleware"

    @staticmethod
    def _unsafe_execute_reason(command: str) -> str | None:
        if not command:
            return None
        if "\n" in command and "<<" not in command and any(x in command for x in ('"', "`", "$(")):
            return (
                "multi-line content is embedded in a shell string. Use a structured "
                "write/edit tool or a single-quoted heredoc instead."
            )
        if re.search(r"\bpython3?\s+-c\s+(['\"]).*;\s*(for|if|while|def|class|with)\b", command, re.S):
            return (
                "python -c one-liner contains a statement after ';'. Write a script "
                "file or use a heredoc instead."
            )
        return None

    def wrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name") or getattr(getattr(request, "tool", None), "name", "")
        if tool_name != "execute":
            return handler(request)
        args = tool_call.get("args", {}) or {}
        command = args.get("command", "")
        reason = self._unsafe_execute_reason(command if isinstance(command, str) else str(command))
        if not reason:
            return handler(request)
        return ToolMessage(
            content=(
                "[SHELL-SAFETY] blocked unsafe execute command: "
                f"{reason} Do not retry the same command shape."
            ),
            tool_call_id=tool_call.get("id", ""),
            name="execute",
        )

    async def awrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name") or getattr(getattr(request, "tool", None), "name", "")
        if tool_name != "execute":
            return await handler(request)
        args = tool_call.get("args", {}) or {}
        command = args.get("command", "")
        reason = self._unsafe_execute_reason(command if isinstance(command, str) else str(command))
        if not reason:
            return await handler(request)
        return ToolMessage(
            content=(
                "[SHELL-SAFETY] blocked unsafe execute command: "
                f"{reason} Do not retry the same command shape."
            ),
            tool_call_id=tool_call.get("id", ""),
            name="execute",
        )


class PathNormalizerMiddleware(AgentMiddleware):
    """Strip leading '/' from glob/grep results so the agent sees relative paths.

    In virtual_mode=True the workspace root is '/', so tools return paths like
    '/src/foo.py'.  The agent often copies these verbatim into output files,
    failing tasks that expect 'src/foo.py'.  This middleware rewrites tool
    results to use relative paths, which the agent can both use with tools
    (they still resolve correctly) and write into output files without issues.
    """

    name = "PathNormalizerMiddleware"

    _PATH_TOOLS = {"glob", "grep"}

    @staticmethod
    def _strip_leading_slash(text: str) -> str:
        lines = text.split("\n")
        out = []
        for line in lines:
            stripped = re.sub(r"(?<![:\w])/(?=\w)", "", line, count=1)
            out.append(stripped)
        return "\n".join(out)

    def wrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name") or getattr(getattr(request, "tool", None), "name", "")
        result = handler(request)
        if tool_name in self._PATH_TOOLS and isinstance(result.content, str):
            result = ToolMessage(
                content=self._strip_leading_slash(result.content),
                tool_call_id=result.tool_call_id,
                name=result.name,
            )
        return result

    async def awrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name") or getattr(getattr(request, "tool", None), "name", "")
        result = await handler(request)
        if tool_name in self._PATH_TOOLS and isinstance(result.content, str):
            result = ToolMessage(
                content=self._strip_leading_slash(result.content),
                tool_call_id=result.tool_call_id,
                name=result.name,
            )
        return result


class ContextWindowGuardMiddleware(AgentMiddleware):
    """Trigger Deep Agents compaction before an unprofiled GigaChat overflows.

    Deep Agents derives fractional summarization thresholds from
    ``model.profile["max_input_tokens"]``.  Current ``langchain-gigachat``
    models leave that profile unset, so the stock middleware falls back to an
    absolute 170k-token trigger, beyond GigaChat's 128k context window.

    This middleware runs inside the stock Deep Agents summarizer.  When an
    unprofiled request for a known model reaches the configured safe fraction,
    it raises the standard ``ContextOverflowError``. Model context windows come
    from ``GIGACHAT_CONTEXT_WINDOWS`` unless explicitly overridden. Unknown
    models only use provider-error translation. The outer summarizer handles
    either signal through its normal history-offload, summary, and retry path.
    """

    name = "ContextWindowGuardMiddleware"
    _CONTEXT_ERROR_CLASS_NAMES = {
        "RequestEntityTooLargeError",
    }
    _CONTEXT_ERROR_TEXT = (
        "context length",
        "context limit",
        "context window",
        "maximum context",
        "payload too large",
        "too many tokens",
    )

    def __init__(
        self,
        *,
        context_window: int | None = None,
        trigger_fraction: float = 0.85,
        minimum_messages: int = 7,
    ) -> None:
        if context_window is not None and context_window <= 0:
            msg = "context_window must be a positive integer"
            raise ValueError(msg)
        if not 0 < trigger_fraction < 1:
            msg = "trigger_fraction must be between 0 and 1"
            raise ValueError(msg)
        if minimum_messages < 2:
            msg = "minimum_messages must be at least 2"
            raise ValueError(msg)
        self.context_window = context_window
        self.trigger_fraction = trigger_fraction
        self.minimum_messages = minimum_messages
        self.trigger_tokens = (
            int(context_window * trigger_fraction)
            if context_window is not None
            else None
        )

    @staticmethod
    def _model_has_context_profile(model: Any) -> bool:
        profile = getattr(model, "profile", None)
        return (
            isinstance(profile, dict)
            and isinstance(profile.get("max_input_tokens"), int)
            and profile["max_input_tokens"] > 0
        )

    @staticmethod
    def _model_identifier(model: Any) -> str:
        identifier = (
            getattr(model, "model", None)
            or getattr(model, "model_name", None)
            or "GigaChat"
        )
        normalized = str(identifier).strip()
        parts = normalized.split(":")
        if len(parts) > 1 and parts[0].casefold() in {"giga", "gigachat"}:
            normalized = parts[1]
        else:
            normalized = parts[0]
        if normalized.casefold().endswith("-preview"):
            normalized = normalized[: -len("-preview")]
        return normalized

    def context_window_for_model(self, model: Any) -> int | None:
        """Resolve explicit override first, then the known-model table."""
        if self.context_window is not None:
            return self.context_window
        identifier = self._model_identifier(model)
        return _NORMALIZED_CONTEXT_WINDOWS.get(identifier.casefold())

    @classmethod
    def _is_provider_context_overflow(cls, exc: Exception) -> bool:
        if type(exc).__name__ in cls._CONTEXT_ERROR_CLASS_NAMES:
            return True
        if type(exc).__name__ not in {"BadRequestError", "UnprocessableEntityError"}:
            return False
        text = str(exc).lower()
        return any(marker in text for marker in cls._CONTEXT_ERROR_TEXT)

    @staticmethod
    def _request_tokens(request: Any) -> int:
        messages = list(getattr(request, "messages", []) or [])
        system_message = getattr(request, "system_message", None)
        if system_message is not None:
            messages.insert(0, system_message)
        return count_tokens_approximately(
            messages,
            tools=getattr(request, "tools", None),
            use_usage_metadata_scaling=True,
        )

    @staticmethod
    def _is_summarization_retry(request: Any) -> bool:
        """Return whether Deep Agents just compacted and is retrying the model.

        The stock summarizer calls the inner handler once more with a new
        synthetic summary before its state update is committed.  Raising a
        second overflow from that retry would escape the outer catch block.
        """
        messages = list(getattr(request, "messages", []) or [])
        if not messages:
            return False
        first = messages[0]
        additional_kwargs = getattr(first, "additional_kwargs", {}) or {}
        if additional_kwargs.get("lc_source") != "summarization":
            return False
        state = getattr(request, "state", {}) or {}
        event = state.get("_summarization_event") if isinstance(state, dict) else None
        if not isinstance(event, dict):
            return True
        previous_summary = event.get("summary_message")
        return getattr(previous_summary, "content", None) != getattr(
            first, "content", None
        )

    def _raise_if_near_limit(self, request: Any) -> None:
        messages = list(getattr(request, "messages", []) or [])
        if (
            self._model_has_context_profile(getattr(request, "model", None))
            or self._is_summarization_retry(request)
            or len(messages) < self.minimum_messages
        ):
            return
        context_window = self.context_window_for_model(
            getattr(request, "model", None)
        )
        if context_window is None:
            return
        trigger_tokens = int(context_window * self.trigger_fraction)
        total_tokens = self._request_tokens(request)
        if total_tokens < trigger_tokens:
            return
        raise ContextOverflowError(
            "GigaChat context guard requested proactive compaction at "
            f"{total_tokens} estimated tokens "
            f"({self.trigger_fraction:.0%} of {context_window})."
        )

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        self._raise_if_near_limit(request)
        try:
            return handler(request)
        except Exception as exc:
            if not self._is_provider_context_overflow(exc):
                raise
            raise ContextOverflowError(
                "GigaChat rejected the request because its context window was exceeded."
            ) from exc

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self._raise_if_near_limit(request)
        try:
            return await handler(request)
        except Exception as exc:
            if not self._is_provider_context_overflow(exc):
                raise
            raise ContextOverflowError(
                "GigaChat rejected the request because its context window was exceeded."
            ) from exc


class DeterministicOutputMiddleware(AgentMiddleware):
    """Prevent ungrounded direct writes for derived data outputs.

    The guard is deliberately structural rather than task-specific. It only
    applies when a new scalar/structured output is being written in a workspace
    that already contains source files, and no deterministic computation tool
    has run. Source-code deliverables and edits to existing files are untouched.
    """

    name = "DeterministicOutputMiddleware"
    _DECISION_MARKER = "[DETERMINISTIC-NEXT-STEP]"
    _OBSERVATION_TOOLS = {"glob", "grep", "ls", "read_file"}
    _STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl", ".tsv"}
    _SOURCE_SUFFIXES = {
        ".csv",
        ".db",
        ".json",
        ".jsonl",
        ".log",
        ".sqlite",
        ".tsv",
        ".txt",
        ".xlsx",
        ".xml",
    }
    _CODE_SUFFIXES = {".awk", ".js", ".pl", ".py", ".rb", ".sh", ".sql", ".ts"}
    _DETERMINISTIC_COMMAND = re.compile(
        r"\b(?:awk|jq|python|python3|sed|sha256sum|sqlite3|wc)\b"
    )

    @staticmethod
    def _is_skill_workspace() -> bool:
        workspace = get_workspace_path()
        return (
            workspace is not None
            and (workspace / ".agents" / "skills").is_dir()
        )

    @staticmethod
    def _tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
        return [
            call
            for message in messages
            if isinstance(message, AIMessage)
            for call in (getattr(message, "tool_calls", None) or [])
        ]

    @classmethod
    def _has_deterministic_call(cls, messages: list[Any]) -> bool:
        for call in cls._tool_calls(messages):
            if call.get("name") == "execute":
                command = str((call.get("args") or {}).get("command", ""))
                if cls._DETERMINISTIC_COMMAND.search(command):
                    return True
        return False

    @classmethod
    def _requires_computation(cls, workspace: Path, path: str, content: str) -> bool:
        target = workspace / path.lstrip("/")
        if target.exists() or target.suffix.lower() in cls._CODE_SUFFIXES:
            return False
        try:
            existing_files = [item for item in workspace.rglob("*") if item.is_file()]
        except OSError:
            return False
        if not existing_files:
            return False
        if re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", content):
            return True
        if target.suffix.lower() not in cls._STRUCTURED_SUFFIXES:
            return False
        return any(
            item.suffix.lower() in cls._SOURCE_SUFFIXES and item != target
            for item in existing_files
        )

    def _blocked_result(self, request: Any) -> ToolMessage | None:
        if self._is_skill_workspace():
            return None
        tool_call = getattr(request, "tool_call", {}) or {}
        if tool_call.get("name") != "write_file":
            return None
        args = tool_call.get("args", {}) or {}
        path = str(args.get("file_path") or args.get("path") or "")
        content = str(args.get("content") or "")
        workspace = get_workspace_path()
        state = getattr(request, "state", {}) or {}
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if (
            not path
            or workspace is None
            or self._has_deterministic_call(messages)
            or not self._requires_computation(workspace, path, content)
        ):
            return None
        return ToolMessage(
            content=(
                "[DETERMINISTIC-OUTPUT] Directly writing a guessed/mentally computed "
                "derived value is blocked. Use `execute` with a single-quoted Python 3 "
                "heredoc (`python3 <<'PY' ... PY`) to read the actual workspace inputs and "
                "create the requested output deterministically. Do not create a helper "
                "script inside the workspace because it may contaminate recursive counts, "
                "manifests, or searches."
            ),
            tool_call_id=tool_call.get("id", ""),
            name="write_file",
        )

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        if self._is_skill_workspace():
            return None
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages or LoopBreakerMiddleware._already_nudged(
            messages, self._DECISION_MARKER
        ):
            return None
        calls = self._tool_calls(messages)
        if not calls or self._has_deterministic_call(messages):
            return None
        last_call = calls[-1]
        if last_call.get("name") not in self._OBSERVATION_TOOLS:
            return None
        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"{self._DECISION_MARKER} Decide how the inspected content affects "
                        "the requested result. If the result requires counting, filtering, "
                        "parsing, aggregation, hashing, ordering, or another derived "
                        "transformation, your NEXT tool must be `execute` with a single-quoted "
                        "Python 3 heredoc (`python3 <<'PY' ... PY`); have that code read the "
                        "real workspace files and write the final output. Do not compute or "
                        "transcribe derived values mentally. For a multi-rule transformation "
                        "or code, first use `think` to enumerate the exact output fields and "
                        "types, ordering/tie-break rules, missing-value conventions, and every "
                        "edge-case branch from the user's request. If the task is only a "
                        "literal surgical text edit, proceed with `edit_file` instead."
                    )
                )
            ]
        }

    def wrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        return self._blocked_result(request) or handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> ToolMessage:
        return self._blocked_result(request) or await handler(request)


class SpecificationAuditMiddleware(AgentMiddleware):
    """Audit newly generated outputs against the user's explicit contract.

    The middleware records the files present when the runner selects a workspace.
    When the agent later reads a newly created file, it gets one concise audit
    pass focused on schema fidelity. This is task-agnostic: no benchmark names,
    filenames, expected values, or domain rules are encoded here.
    """

    name = "SpecificationAuditMiddleware"
    _MARKER = "[SPECIFICATION-AUDIT]"

    @staticmethod
    def _last_tool_call(messages: list[Any]) -> dict[str, Any] | None:
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            calls = getattr(message, "tool_calls", None) or []
            if calls:
                return calls[-1]
        return None

    @staticmethod
    def _read_path(call: dict[str, Any]) -> str:
        args = call.get("args", {}) or {}
        return str(args.get("file_path") or args.get("path") or "").lstrip("/")

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages or LoopBreakerMiddleware._already_nudged(messages, self._MARKER):
            return None
        call = self._last_tool_call(messages)
        if not call or call.get("name") != "read_file":
            return None
        path = self._read_path(call)
        if not path or path in get_initial_workspace_files():
            return None
        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"{self._MARKER} You just inspected the generated output `{path}`. "
                        "Before finishing, compare it field-by-field with the user's original "
                        "contract; do not merely declare it correct.\n"
                        "- Exact filename, field/column/key names, nesting, and no extra fields.\n"
                        "- Exact value types: number vs string vs null vs empty string vs list/object.\n"
                        "- Required ordering, stable input order, tie-breakers, and deduplication rules.\n"
                        "- Every named status/branch and all positive, negative, missing, retry, "
                        "and boundary cases in the request.\n"
                        "- For code, execute focused examples covering every stated branch and "
                        "fix the implementation if any result differs.\n"
                        "If any requirement is not visibly satisfied, fix the output now and "
                        "read it once more. Otherwise finish."
                    )
                )
            ]
        }


class MemoryTaskMiddleware(AgentMiddleware):
    """Nudge the agent on memory-task workflows when AGENTS.md is present."""

    name = "MemoryTaskMiddleware"
    _START_MARKER = "[MEMORY-TASK]"
    _SAVE_MARKER = "[MEMORY-SAVE]"

    @staticmethod
    def _is_memory_workspace() -> bool:
        wp = get_workspace_path()
        return wp is not None and (wp / "AGENTS.md").exists()

    @staticmethod
    def _messages_text(messages: list[Any]) -> str:
        parts: list[str] = []
        for msg in messages:
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)

    @staticmethod
    def _memory_touched(messages: list[Any]) -> bool:
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for tc in getattr(msg, "tool_calls", None) or []:
                args = tc.get("args", {}) or {}
                for key in ("file_path", "path", "target_file"):
                    val = args.get(key, "")
                    if isinstance(val, str) and "MEMORY.md" in val:
                        return True
        return False

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        if not self._is_memory_workspace():
            return None
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None
        text = self._messages_text(messages)

        ai_rounds = sum(
            1 for m in messages if isinstance(m, AIMessage) and (getattr(m, "tool_calls", None) or [])
        )
        if ai_rounds == 0 and self._START_MARKER not in text:
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            f"{self._START_MARKER} This is a memory task.\n"
                            "1) read_file MEMORY.md first\n"
                            "2) complete the requested deliverable using MEMORY facts verbatim\n"
                            "3) if the user mentioned ANY personal fact, edit_file MEMORY.md before finishing"
                        )
                    )
                ]
            }

        if (
            ai_rounds >= 2
            and not self._memory_touched(messages)
            and self._SAVE_MARKER not in text
        ):
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            f"{self._SAVE_MARKER} Did the user mention personal facts "
                            "(city, provider, focus day, name, tools)? If yes, you MUST "
                            "edit_file MEMORY.md NOW before finishing. This is mandatory."
                        )
                    )
                ]
            }
        return None


class LoopBreakerMiddleware(AgentMiddleware):
    """Detect agent loops and repeated failed command families.

    GigaChat-3-Ultra on deepagents 0.6.x occasionally commits to a broken
    pattern on turn 1 (e.g. one-line `python -c "...; for v in xs: s += v; ..."`
    which is a SyntaxError) and then retries the exact same call until the
    recursion-limit kicks in. The agent normally has 80 steps and burns all
    of them on this loop, dropping the task. The path-fix and SyntaxError
    advice in the system prompt help, but on hot paths the model still
    falls into this pattern on a fraction of runs.

    This middleware watches `messages` in `before_model` and, when it sees
    the same `(tool_name, args)` for 3 consecutive AIMessages, appends a
    one-shot SystemMessage with a forceful instruction to STOP and switch
    strategy. The model usually breaks out of the loop on the next turn.
    """

    name = "LoopBreakerMiddleware"

    def _last_n_tool_pairs(
        self, messages: list[Any], n: int
    ) -> list[tuple[str, str, str]] | None:
        """Return last n consecutive (tool_name, args_json, result_text) tuples.

        Walks back over the messages stripping AIMessage + matching ToolMessage
        pairs. Returns None as soon as the chain breaks (e.g. an AIMessage with
        no tool_calls, a HumanMessage, etc.).
        """
        pairs: list[tuple[str, str, str]] = []
        i = len(messages) - 1
        while i >= 0 and len(pairs) < n:
            msg = messages[i]
            if isinstance(msg, ToolMessage):
                # Find the matching AIMessage immediately before it.
                if i == 0:
                    return None
                ai = messages[i - 1]
                if not isinstance(ai, AIMessage):
                    return None
                tcs = getattr(ai, "tool_calls", None) or []
                if not tcs:
                    return None
                tc = tcs[0]
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                pairs.append(
                    (
                        tc.get("name", ""),
                        json.dumps(tc.get("args", {}), sort_keys=True),
                        content,
                    )
                )
                i -= 2
            elif isinstance(msg, AIMessage):
                # Trailing assistant message with no tool result yet — skip.
                i -= 1
            else:
                # SystemMessage / HumanMessage — chain broken.
                break
        return pairs if len(pairs) == n else None

    @staticmethod
    def _result_is_error(text: str) -> bool:
        """Heuristic: does this tool result look like a failure?"""
        if not text:
            return False
        markers = (
            "Error:",
            "error:",
            "Cannot ",
            "cannot ",
            "Traceback",
            "[stderr]",
            "Exit code: 1",
            "Exit code: 2",
            "SyntaxError",
            "FileNotFoundError",
            "No such file",
            "String not found",
            "Read-only file system",
            "unrecognized arguments",
            "invalid choice",
            "unknown command",
            "command not found",
            "[SHELL-SAFETY]",
        )
        return any(m in text for m in markers)

    @staticmethod
    def _error_family(text: str) -> str | None:
        lowered = text.lower()
        families = (
            ("invalid-cli-args", ("unrecognized arguments", "invalid choice", "unknown command")),
            ("shell-command-not-found", ("command not found",)),
            ("python-syntax", ("syntaxerror",)),
            ("missing-path", ("no such file", "read-only file system")),
            ("edit-miss", ("string not found",)),
            ("shell-safety", ("[shell-safety]",)),
            ("traceback", ("traceback",)),
        )
        for family, markers in families:
            if any(marker in lowered for marker in markers):
                return family
        return None

    @staticmethod
    def _count_tool_rounds(messages: list[Any]) -> int:
        return sum(
            1
            for msg in messages
            if isinstance(msg, AIMessage) and (getattr(msg, "tool_calls", None) or [])
        )

    @staticmethod
    def _grep_looks_empty(text: str) -> bool:
        if not text or not text.strip():
            return True
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in ("no matches", "0 matches", "not found", "no results", "0 results")
        )

    @staticmethod
    def _already_nudged(messages: list[Any], marker: str) -> bool:
        for msg in messages:
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and marker in content:
                return True
        return False

    def _budget_nudge(self, tool_rounds: int, *, final: bool = False) -> str:
        marker = "[BUDGET-NUDGE-FINAL]" if final else "[BUDGET-NUDGE-BATCH]"
        if final:
            return (
                f"{marker} You have made {tool_rounds} tool calls. Complete the remaining "
                "deliverables now, run one focused verification, fix any reported failure, "
                "and finish. Do not restart exploration or repeat a failed call shape."
            )
        return (
            f"{marker} You have made {tool_rounds} tool calls. Preserve completed work, "
            "but switch to a bounded batch strategy now.\n"
            "- Use one script/command for repeated edits, parsing, or ordered operations.\n"
            "- Do not retry the same tool/error shape.\n"
            "- Then verify the requested outputs once and finish."
        )

    def _grep_empty_nudge(self) -> str:
        return (
            "[LOOP-BREAKER] grep returned 0 matches twice. Do NOT grep again.\n"
            "Switch to one `execute` call using `python3 <<'PY' ... PY`, scan the "
            "right directory with pathlib.rglob, and write the requested final output. "
            "Do not create a helper file inside the scanned workspace."
        )

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None

        tool_rounds = self._count_tool_rounds(messages)

        # Staged call-count guard. Each marker is injected at most once. The
        # previous implementation only searched back to the latest AIMessage,
        # so it re-injected the same nudge on every subsequent model turn and
        # could itself keep a task in a loop.
        if tool_rounds >= 24 and not self._already_nudged(
            messages, "[BUDGET-NUDGE-FINAL]"
        ):
            return {
                "messages": [
                    HumanMessage(content=self._budget_nudge(tool_rounds, final=True))
                ]
            }
        if tool_rounds >= 12 and not self._already_nudged(
            messages, "[BUDGET-NUDGE-BATCH]"
        ):
            return {"messages": [HumanMessage(content=self._budget_nudge(tool_rounds))]}

        # Grep-empty streak: common on count tasks that return 0 forever.
        grep_pairs = self._last_n_tool_pairs(messages, 2)
        if (
            grep_pairs
            and all(p[0] == "grep" for p in grep_pairs)
            and all(self._grep_looks_empty(p[2]) for p in grep_pairs)
            and not self._already_nudged(messages, "[LOOP-BREAKER]")
        ):
            return {"messages": [HumanMessage(content=self._grep_empty_nudge())]}

        pairs = self._last_n_tool_pairs(messages, 3)
        if not pairs:
            return None

        # Two trigger conditions:
        # 1. Exact same (tool, args) 3 times (original behavior).
        # 2. Same tool returning error-like results 3 times in a row, even if
        #    args differ slightly — common when the model keeps tweaking a
        #    broken pattern without fixing the real bug (e.g. <n>\t prefix
        #    leak in edit_file).
        names = {p[0] for p in pairs}
        all_same_call = pairs[0] == pairs[1] == pairs[2]
        all_same_tool_errors = (
            len(names) == 1
            and all(self._result_is_error(p[2]) for p in pairs)
        )
        families = [self._error_family(p[2]) for p in pairs]
        all_same_error_family = bool(families[0]) and families[0] == families[1] == families[2]
        if not (all_same_call or all_same_tool_errors or all_same_error_family):
            return None

        # Avoid injecting the same nudge more than once in a run.
        already_injected_marker = "[LOOP-BREAKER]"
        if self._already_nudged(messages, already_injected_marker):
            return None
        tool_name = pairs[0][0]
        last_result = pairs[0][2][:300]
        nudge = (
            f"{already_injected_marker} You have called `{tool_name}` 3 times "
            f"in a row and it keeps failing (last error: {last_result!r}). "
            f"STOP repeating this approach. Change strategy:\n"
            f"- If `edit_file` says 'String not found' and the text came from "
            f"`read_file`: you are leaking the leading '<line_no>\\t' prefix. "
            f"Strip the spaces + number + tab before reusing the text. "
            f"`     3\\tHello` in display means the file contains just `Hello`.\n"
            f"- If `python3 -c \"...\"` keeps giving SyntaxError: switch to one "
            f"`execute` call with a single-quoted Python 3 heredoc.\n"
            f"- If a filesystem tool path failed, use a relative path like "
            f"`foo.py` or `src/foo.py`. Do NOT use absolute paths.\n"
            f"- If `grep`/`glob` returns nothing useful: for count tasks, write "
            f"a Python 3 heredoc with pathlib.rglob — do NOT keep grepping.\n"
            f"- If `write_file` says 'already exists': the right tool is "
            f"`edit_file`, NOT another `write_file` with a new name.\n"
            f"- If a CLI/runtime tool says 'invalid choice' or 'unrecognized "
            f"arguments': stop inventing flags or subcommands. Use only the "
            f"tool contract that was supplied for this run.\n"
            f"- If shell safety blocked a command: use structured file/runtime "
            f"tools or a single-quoted heredoc; do not embed multi-line content "
            f"in a quoted shell string.\n"
            f"Do something materially different on the next step."
        )
        # IMPORTANT: GigaChat enforces "system message must be the first
        # message" — injecting a mid-conversation SystemMessage causes a
        # hard 400 BadRequest. Send the nudge as a HumanMessage instead;
        # the model still follows it, and GigaChat accepts the shape.
        return {"messages": [HumanMessage(content=nudge)]}


def _tool_description_overrides(profile_variant: str) -> dict[str, str]:
    if profile_variant not in {"native", "native_fs", "filesystem", "fs"}:
        return {
            "execute": (
                "Run one short shell command only when the active runtime tool "
                "contract permits shell use. Prefer structured tools for reading "
                "or writing content. Never embed multi-line content in a "
                "double-quoted shell string."
            )
        }
    return {
        "ls": (
            "List files in a directory. Use relative paths: `ls .` or "
            "`ls src`. Do NOT use absolute paths like '/Users/name/project'."
        ),
        "read_file": (
            "Read a file. Use a relative path like 'foo.py' or "
            "'src/foo.py'. Do NOT use absolute paths. "
            "Output is prefixed with '<line_no>\\t' for display — "
            "strip that prefix before reusing the text in edit_file/write_file."
        ),
        "glob": (
            "Find files by pattern (e.g. '**/*.py'). Returns paths "
            "starting with '/' (virtual root). IMPORTANT: when you write "
            "these paths to any output file, strip the leading '/' — write "
            "'src/foo.py', NOT '/src/foo.py'."
        ),
        "write_file": (
            "Create a file or overwrite it completely. Use a relative "
            "path like 'foo.py' or 'src/foo.py'. The content is the file "
            "body verbatim — do NOT include line-number prefixes from "
            "read_file output. Use this for new files or full rewrites; use "
            "edit_file for small changes. When the task names a required "
            "output file, write the final deliverable content into that exact "
            "file (do NOT write a script as a substitute). Unless explicitly "
            "requested, do not leave the file empty or with placeholder text."
        ),
        "edit_file": (
            "Replace one exact occurrence of old_string with new_string in "
            "an existing file. **CRITICAL: STRIP the leading '<line_no>\\t' "
            "prefix from read_file output before putting text into "
            "old_string or new_string.** Example: read_file shows "
            "`     3\\tHello world` — you must pass `old_string='Hello "
            "world'`, NOT `old_string='     3\\tHello world'`. The "
            "spaces + line-number + tab prefix is display only, the file "
            "itself does not contain them. If edit_file says 'String not "
            "found' and you copied recently from read_file, the prefix "
            "leak is almost certainly the cause — strip it and retry. "
            "Always include enough surrounding lines so old_string is "
            "unique. Use a relative path like 'foo.py' or 'src/foo.py'."
        ),
        "grep": (
            "Search for a literal substring (NOT a regex) across files. "
            "Pass exactly ONE phrase per call. **Always pass `path`** to scope "
            "the search: path='tests' for tests/*.py, path='src' for src/*.py, "
            "path='.' for the whole workspace. To search several alternatives "
            "run grep several times. The result lists matching lines — read "
            "it directly. If 0 matches, use `execute` with a single-quoted "
            "Python 3 heredoc and pathlib.rglob instead of retrying grep. "
            "Returned paths start with "
            "'/' — strip it before writing to output files."
        ),
        "execute": (
            "Run one short shell command in the workspace directory "
            "(e.g. 'rm a.txt', 'mv old new', 'mkdir -p logs'). "
            "IMPORTANT: use RELATIVE paths — `cat numbers.txt` works, "
            "`cat /numbers.txt` will fail with 'No such file' or "
            "'Read-only file system'. Never embed multi-line content via "
            "sh -c \"...\" or bash -c \"...\" with double quotes; if you "
            "must run a multi-line snippet, use a single-quoted heredoc "
            "(cat <<'EOF' ... EOF). Prefer write_file / edit_file for "
            "changing file content."
        ),
    }


_workspace_path: ContextVar[Path | None] = ContextVar(
    "deepagents_gigachat_workspace_path",
    default=None,
)
_initial_workspace_files: ContextVar[frozenset[str]] = ContextVar(
    "deepagents_gigachat_initial_workspace_files",
    default=frozenset(),
)


def set_workspace_path(path: Path | str | None) -> None:
    """Store the current task workspace in the current execution context.

    Benchmark runners build and invoke agents in worker threads. A process-wide
    global lets concurrent tasks overwrite each other's workspace, so middleware
    could inspect the wrong AGENTS.md or fixture. ContextVar keeps the public
    runner API while isolating each worker context.
    """
    workspace = Path(path) if path is not None else None
    _workspace_path.set(workspace)
    if workspace is None:
        _initial_workspace_files.set(frozenset())
        return
    try:
        files = frozenset(
            item.relative_to(workspace).as_posix()
            for item in workspace.rglob("*")
            if item.is_file()
        )
    except OSError:
        files = frozenset()
    _initial_workspace_files.set(files)


def get_workspace_path() -> Path | None:
    """Return the workspace path set by the runner, if any."""
    return _workspace_path.get()


def get_initial_workspace_files() -> frozenset[str]:
    """Return workspace-relative files captured when the task was initialized."""
    return _initial_workspace_files.get()


def register_harness(
    profile_variant: str | None = None,
    tool_contract: str | None = None,
    *,
    context_window: int | None = None,
    summarization_trigger: float | None = None,
) -> None:
    """Register the GigaChat HarnessProfile under GigaChat provider keys."""
    variant = (profile_variant or os.getenv("DEEPAGENTS_GIGACHAT_PROFILE") or "native_fs").strip().lower().replace("-", "_")
    contract = tool_contract or os.getenv("DEEPAGENTS_GIGACHAT_TOOL_CONTRACT")
    window = context_window
    if window is None:
        configured_window = os.getenv("DEEPAGENTS_GIGACHAT_CONTEXT_WINDOW")
        window = int(configured_window) if configured_window else None
    trigger = summarization_trigger
    if trigger is None:
        trigger = float(
            os.getenv("DEEPAGENTS_GIGACHAT_SUMMARIZATION_TRIGGER", "0.85")
        )
    middleware: list[AgentMiddleware] = [
        ThinkToolMiddleware(),
        ShellSafetyMiddleware(),
        PathNormalizerMiddleware(),
        ContextWindowGuardMiddleware(
            context_window=window,
            trigger_fraction=trigger,
        ),
        DeterministicOutputMiddleware(),
        SpecificationAuditMiddleware(),
        MemoryTaskMiddleware(),
        LoopBreakerMiddleware(),
    ]
    if contract:
        middleware.append(ToolContractMiddleware(contract))

    profile = HarnessProfile(
        base_system_prompt=f"{build_system_prompt(variant)}\n\n",
        tool_description_overrides=_tool_description_overrides(variant),
        extra_middleware=tuple(middleware),
    )

    for provider_key in ("gigachat", "giga"):
        register_harness_profile(provider_key, profile)
    print(
        "[deepagents-gigachat] Harness profile loaded "
        f"(providers=gigachat,giga; variant={variant}; "
        f"context_window={window if window is not None else 'model-map'}; "
        f"summarization_trigger={trigger:.0%}; "
        f"tool_contract={'on' if bool(contract) else 'off'})"
    )
