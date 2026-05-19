"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

import json
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

from deepagents_gigachat.prompts import BASE_SYSTEM_PROMPT


@tool("think")
def _think(thought: str = Field(..., description="A thought to think about.")) -> str:
    """Use this tool as scratchpad to structure intermediate reasoning."""
    return thought


class ThinkToolMiddleware(AgentMiddleware):
    """Inject the local `think` tool into the default toolset."""

    tools = [_think]


class LoopBreakerMiddleware(AgentMiddleware):
    """Detect agent loops (same tool call repeated 3+ times in a row).

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
        )
        return any(m in text for m in markers)

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None
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
        if not (all_same_call or all_same_tool_errors):
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
            f"`write_file run.py \"<multi-line code>\"` then `execute python "
            f"run.py`.\n"
            f"- If a path failed with 'No such file' or 'Read-only file "
            f"system': switch from absolute `/foo` to relative `foo`.\n"
            f"- If `grep`/`glob` returns nothing useful: try a broader search "
            f"term or use `ls` to verify the structure.\n"
            f"- If `write_file` says 'already exists': the right tool is "
            f"`edit_file`, NOT another `write_file` with a new name.\n"
            f"Do something materially different on the next step."
        )
        # IMPORTANT: GigaChat enforces "system message must be the first
        # message" — injecting a mid-conversation SystemMessage causes a
        # hard 400 BadRequest. Send the nudge as a HumanMessage instead;
        # the model still follows it, and GigaChat accepts the shape.
        return {"messages": [HumanMessage(content=nudge)]}


def register_harness() -> None:
    """Register the GigaChat HarnessProfile under GigaChat provider keys."""
    profile = HarnessProfile(
        base_system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n",
        tool_description_overrides={
            # Filesystem tools (deepagents 0.6.x) — explicit relative-path
            # rule so they match `execute` semantics (host shell, not virtual).
            "ls": (
                "List files in a directory. Use a relative path (e.g. '.', "
                "'src') — NEVER absolute '/'."
            ),
            "read_file": (
                "Read a file. Use a relative path like 'foo.py' (NEVER start "
                "with '/'). Output is prefixed with '<line_no>\\t' for "
                "display — strip that prefix before reusing the text in "
                "edit_file/write_file."
            ),
            "glob": (
                "Find files by pattern (e.g. '**/*.py'). Patterns are "
                "relative to the workspace; do NOT prefix with '/'."
            ),
            "write_file": (
                "Create a file or overwrite it completely. Use a relative path "
                "like 'foo.py' or 'src/foo.py' (never start with '/'). The "
                "content is the file body verbatim — do NOT include line-number "
                "prefixes from read_file output. Use this for new files or full "
                "rewrites; use edit_file for small changes. When the task names "
                "a required output file, write the final deliverable content into "
                "that exact file (do NOT write a script as a substitute). Unless "
                "explicitly requested, do not leave the file empty or with "
                "placeholder text."
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
                "unique. Use a relative path (never start with '/')."
            ),
            "grep": (
                "Search for a literal substring (NOT a regex) across files. "
                "Pass exactly ONE phrase per call. To search for several "
                "alternatives run grep several times. The result lists matching "
                "lines — read it directly instead of opening every matched "
                "file again."
            ),
            "execute": (
                "Run one short shell command in the workspace directory "
                "(e.g. 'rm a.txt', 'mv old new', 'mkdir -p logs'). "
                "IMPORTANT: this runs on the host filesystem, NOT the virtual "
                "root used by the file tools. Use RELATIVE paths — `cat "
                "numbers.txt` works, `cat /numbers.txt` will fail with "
                "'No such file' or 'Read-only file system' (it would read "
                "the real /). Never embed multi-line content via sh -c \"...\" "
                "or bash -c \"...\" with double quotes; if you must run a "
                "multi-line snippet, use a single-quoted heredoc "
                "(cat <<'EOF' ... EOF). Prefer write_file / edit_file for "
                "changing file content. Use execute for quick verification "
                "of required outputs (e.g., `ls -l`, `wc -l`) before finishing."
            ),
        },
        extra_middleware=(ThinkToolMiddleware(), LoopBreakerMiddleware()),
    )

    for provider_key in ("gigachat", "giga"):
        register_harness_profile(provider_key, profile)
