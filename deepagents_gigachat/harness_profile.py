"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

import json
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
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

    def _last_n_tool_calls(self, messages: list[Any], n: int) -> list[tuple[str, str]] | None:
        """Return the last n consecutive AI tool-call signatures, or None."""
        sigs: list[tuple[str, str]] = []
        # walk backwards collecting AI tool calls; tolerate ToolMessage between
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                tcs = getattr(m, "tool_calls", None) or []
                if not tcs:
                    return None
                # GigaChat sends only one tool call per AIMessage
                tc = tcs[0]
                sigs.append((tc.get("name", ""), json.dumps(tc.get("args", {}), sort_keys=True)))
                if len(sigs) >= n:
                    return sigs
            elif isinstance(m, ToolMessage):
                continue
            else:
                # HumanMessage or SystemMessage — stop walking back
                break
        return None

    def before_model(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if not messages:
            return None
        sigs = self._last_n_tool_calls(messages, 3)
        if not sigs or len(sigs) < 3:
            return None
        if not (sigs[0] == sigs[1] == sigs[2]):
            return None
        # Avoid injecting the same nudge twice in a row.
        already_injected_marker = "[LOOP-BREAKER]"
        for m in reversed(messages):
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and already_injected_marker in content:
                return None
            if isinstance(m, AIMessage):
                break
        tool_name, _ = sigs[0]
        nudge = (
            f"{already_injected_marker} You have called `{tool_name}` with the "
            f"same arguments 3 times in a row and the result has been identical "
            f"each time. STOP. Do NOT call `{tool_name}` with those arguments "
            f"again. Change your approach completely:\n"
            f"- If you were using `python -c \"...\"` and got a SyntaxError, "
            f"write the code to a script file with `write_file run.py \"...\"` "
            f"and then `execute python run.py`.\n"
            f"- If a path failed with 'No such file' or 'Read-only file system', "
            f"switch from absolute `/foo` to relative `foo`.\n"
            f"- If `grep`/`glob` is returning nothing useful, try a different "
            f"search term or use `ls`.\n"
            f"Do something materially different on the next step."
        )
        return {"messages": [SystemMessage(content=nudge)]}


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
                "an existing file. Copy old_string verbatim from a prior "
                "read_file (without the leading '<line_no>\\t' prefix), "
                "including blank lines and indentation. Add a few lines of "
                "surrounding context so old_string is unique within the file. "
                "Use a relative path (never start with '/')."
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
        # Drop the TodoListMiddleware (and its `write_todos` tool + 3.6 KB
        # tool description + after_model enforcement) — atomic bench tasks
        # don't benefit from plan-then-act, and deepagents 0.6.x inflated
        # the description from ~1 KB to ~3.6 KB, eating the recursion-limit
        # budget. Also disable the auto-added general-purpose subagent so
        # the `task` tool (6.9 KB description in 0.6.x) disappears too.
        excluded_middleware=frozenset({"TodoListMiddleware"}),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )

    for provider_key in ("gigachat", "giga"):
        register_harness_profile(provider_key, profile)
