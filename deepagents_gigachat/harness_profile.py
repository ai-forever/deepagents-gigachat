"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from deepagents import (
    HarnessProfile,
    register_harness_profile,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from pydantic import Field

from deepagents_gigachat.prompts import build_system_prompt


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
        for msg in reversed(messages):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and marker in content:
                return True
            if isinstance(msg, AIMessage):
                break
        return False

    def _budget_nudge(self, tool_rounds: int) -> str:
        return (
            "[BUDGET-NUDGE] You have made "
            f"{tool_rounds} tool calls. STOP exploring and finish the task NOW.\n"
            "- Simple count task? write_file run.py + execute python run.py → done.\n"
            "- Required output file not written yet? write_file it immediately.\n"
            "- Do NOT make more read/grep calls. Write your best answer and finish."
        )

    def _grep_empty_nudge(self) -> str:
        return (
            "[LOOP-BREAKER] grep returned 0 matches twice. Do NOT grep again.\n"
            "Switch to: write_file run.py using pathlib.Path('tests').rglob('*.py') "
            "(or the right directory), count occurrences in a loop, "
            "open('count.txt','w').write(str(total)), then execute python run.py."
        )

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None

        tool_rounds = self._count_tool_rounds(messages)

        # High call-count guard: force finish on simple-looking tasks.
        if tool_rounds >= 12 and not self._already_nudged(messages, "[BUDGET-NUDGE]"):
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

        # Avoid injecting the same nudge twice in a row.
        already_injected_marker = "[LOOP-BREAKER]"
        for m in reversed(messages):
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and already_injected_marker in content:
                return None
            if isinstance(m, AIMessage):
                break
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
            f"- If `python -c \"...\"` keeps giving SyntaxError: switch to "
            f"`write_file run.py \"<multi-line code>\"`, then `execute python run.py`.\n"
            f"- If a filesystem tool path failed, use a relative path like "
            f"`foo.py` or `src/foo.py`. Do NOT use absolute paths.\n"
            f"- If `grep`/`glob` returns nothing useful: for count tasks, write "
            f"`run.py` with pathlib.rglob — do NOT keep grepping.\n"
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
            "it directly. If 0 matches, switch to a Python run.py script with "
            "pathlib.rglob instead of retrying grep. Returned paths start with "
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


_workspace_path: Path | None = None


def set_workspace_path(path: Path | str) -> None:
    """Store the current task workspace so middleware can locate workspace files."""
    global _workspace_path  # noqa: PLW0603
    _workspace_path = Path(path) if path is not None else None


def get_workspace_path() -> Path | None:
    """Return the workspace path set by the runner, if any."""
    return _workspace_path


def register_harness(profile_variant: str | None = None, tool_contract: str | None = None) -> None:
    """Register the GigaChat HarnessProfile under GigaChat provider keys."""
    variant = (profile_variant or os.getenv("DEEPAGENTS_GIGACHAT_PROFILE") or "native_fs").strip().lower().replace("-", "_")
    contract = tool_contract or os.getenv("DEEPAGENTS_GIGACHAT_TOOL_CONTRACT")
    middleware: list[AgentMiddleware] = [
        ThinkToolMiddleware(),
        ShellSafetyMiddleware(),
        PathNormalizerMiddleware(),
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
        f"tool_contract={'on' if bool(contract) else 'off'})"
    )
