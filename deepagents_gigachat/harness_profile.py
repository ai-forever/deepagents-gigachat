"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from deepagents import HarnessProfile, register_harness_profile
from deepagents.middleware._utils import append_to_system_message
from deepagents.profiles import GeneralPurposeSubagentProfile
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import tool
from pydantic import Field

from deepagents_gigachat.prompts import (
    ADAPTIVE_FILESYSTEM_COMMIT_OVERRIDE,
    ADAPTIVE_SEARCH_POSTPROCESS_OVERRIDE,
    ADAPTIVE_TOOLS_EXECUTE_ONLY_OVERRIDE,
    BASE_SYSTEM_PROMPT,
    HYBRID_LOOP_OVERRIDE,
    PI_LITE_LOOP_OVERRIDE,
    PI_LITE_SYSTEM_PROMPT,
    PI_TOOLS_LOOP_OVERRIDE,
    PI_TOOLS_SYSTEM_PROMPT,
)


@tool("think")
def _think(thought: str = Field(..., description="A thought to think about.")) -> str:
    """Use this tool as scratchpad to structure intermediate reasoning."""
    return thought


class ThinkToolMiddleware(AgentMiddleware):
    """Inject the local `think` tool into the default toolset."""

    tools = [_think]


class PiLiteLoopMiddleware(AgentMiddleware):
    """Append a final pi-like loop reminder after Deep Agents defaults."""

    tools = []

    def _with_loop_override(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        system_message = append_to_system_message(
            request.system_message,
            PI_LITE_LOOP_OVERRIDE,
        )
        return request.override(system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._with_loop_override(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._with_loop_override(request))


class PiToolsLoopMiddleware(PiLiteLoopMiddleware):
    """Append the pi-tools-specific loop reminder."""

    def _with_loop_override(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        system_message = append_to_system_message(
            request.system_message,
            PI_TOOLS_LOOP_OVERRIDE,
        )
        return request.override(system_message=system_message)


class HybridLoopMiddleware(PiLiteLoopMiddleware):
    """Append the hybrid benchmark loop reminder."""

    def _with_loop_override(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        system_message = append_to_system_message(
            request.system_message,
            HYBRID_LOOP_OVERRIDE,
        )
        return request.override(system_message=system_message)


class AdaptiveLoopMiddleware(PiLiteLoopMiddleware):
    """Append generic route-specific loop hints without verifier feedback."""

    def _with_loop_override(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        system_message = append_to_system_message(
            request.system_message,
            HYBRID_LOOP_OVERRIDE,
        )
        prompt = _latest_user_prompt(request)
        for override in _adaptive_route_overrides(prompt):
            system_message = append_to_system_message(system_message, override)
        return request.override(system_message=system_message)


class AdaptiveToolRoutingMiddleware(PiLiteLoopMiddleware):
    """Route data/search/filesystem tasks to a narrower execute-only loop."""

    def _with_loop_override(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        system_message = append_to_system_message(
            request.system_message,
            HYBRID_LOOP_OVERRIDE,
        )
        prompt = _latest_user_prompt(request)
        route = _adaptive_tool_route(prompt)
        if route == "hybrid":
            return request.override(system_message=system_message)

        tools = _filter_tools_by_name(request.tools, {"execute"})
        if not tools:
            return request.override(system_message=system_message)

        system_message = append_to_system_message(
            system_message,
            ADAPTIVE_TOOLS_EXECUTE_ONLY_OVERRIDE,
        )
        for override in _adaptive_tool_route_overrides(route):
            system_message = append_to_system_message(system_message, override)
        return request.override(system_message=system_message, tools=tools)


def _latest_user_prompt(request: ModelRequest[Any]) -> str:
    """Best-effort extraction of the latest user message text."""
    for message in reversed(request.messages):
        message_type = str(getattr(message, "type", "")).lower()
        role = str(getattr(message, "role", "")).lower()
        if message_type in {"human", "user"} or role == "user":
            return _content_to_text(getattr(message, "content", ""))
    if request.messages:
        return _content_to_text(getattr(request.messages[-1], "content", ""))
    return ""


def _content_to_text(content: Any) -> str:
    """Convert common LangChain content shapes into searchable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _adaptive_route_overrides(prompt: str) -> tuple[str, ...]:
    """Return generic adaptive hints for a task prompt."""
    text = f" {prompt.lower()} "
    if _is_filesystem_commit_task(text):
        return (ADAPTIVE_FILESYSTEM_COMMIT_OVERRIDE,)
    if _is_search_postprocess_task(text):
        return (ADAPTIVE_SEARCH_POSTPROCESS_OVERRIDE,)
    return ()


def _adaptive_tool_route(prompt: str) -> str:
    """Return the generic tool-routing bucket for a prompt."""
    text = f" {prompt.lower()} "
    if _is_filesystem_commit_task(text):
        return "filesystem"
    if _is_search_postprocess_task(text):
        return "search"
    if _is_direct_data_task(text):
        return "data"
    return "hybrid"


def _adaptive_tool_route_overrides(route: str) -> tuple[str, ...]:
    """Return concise route hints for execute-only adaptive tool routing."""
    if route == "filesystem":
        return (ADAPTIVE_FILESYSTEM_COMMIT_OVERRIDE,)
    if route == "search":
        return (ADAPTIVE_SEARCH_POSTPROCESS_OVERRIDE,)
    return ()


def _filter_tools_by_name(tools: list[Any], allowed_names: set[str]) -> list[Any]:
    """Keep only tools whose name is in `allowed_names`."""
    return [tool_obj for tool_obj in tools if _tool_name(tool_obj) in allowed_names]


def _tool_name(tool_obj: Any) -> str | None:
    """Best-effort extraction of a tool name from LangChain/dict tool shapes."""
    if isinstance(tool_obj, dict):
        name = tool_obj.get("name")
        if isinstance(name, str):
            return name
        function = tool_obj.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
        return None
    name = getattr(tool_obj, "name", None)
    return name if isinstance(name, str) else None


def _is_direct_data_task(text: str) -> bool:
    data_markers = (
        ".csv",
        ".db",
        ".ini",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".sqlite",
        ".toml",
        ".tsv",
        ".xlsx",
        ".xml",
        ".yaml",
        ".yml",
        " sqlite",
    )
    operation_markers = (
        "aggregate",
        "bucket",
        "convert",
        "count",
        "deduplicate",
        "dedupe",
        "export",
        "extract",
        "filter",
        "group",
        "histogram",
        "join",
        "mean",
        "median",
        "pivot",
        "sort",
        "sum",
        "tally",
        "total",
    )
    return _has_any(text, data_markers) and _has_any(text, operation_markers)


def _is_search_postprocess_task(text: str) -> bool:
    search_markers = (
        "assert",
        "class name",
        "containing",
        "duplicate",
        "email",
        "extract",
        "find",
        "grep",
        "largest",
        "list files",
        "matches",
        "most lines",
        "todo",
    )
    return _has_any(text, search_markers) and (
        _has_any(text, (".py", ".md", ".yaml", ".yml", ".log", " files ", " under "))
    )


def _is_filesystem_commit_task(text: str) -> bool:
    if _has_any(text, (" rename ", " move ", " delete ")):
        return True
    return " convert " in text and _has_any(text, (" package", " directory", " dir "))


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _profile_name() -> str:
    """Return the selected local benchmark profile."""
    return os.getenv("DEEPAGENTS_GIGACHAT_PROFILE", "default").strip().lower()


def _tool_description_overrides() -> dict[str, str]:
    """Tool descriptions tuned for the benchmark harness."""
    return {
        "write_file": (
            "Create a NEW file only. This tool fails if the target path "
            "already exists. Use workspace paths like '/foo.py' or "
            "'/src/foo.py' and never host paths like '/Users/...'. The "
            "content is the file body verbatim — do NOT include line-number "
            "prefixes from read_file output. For existing files, including "
            "full rewrites, use edit_file instead."
        ),
        "edit_file": (
            "Replace one exact occurrence of old_string with new_string in "
            "an existing file. Use this for every change to an existing "
            "file, including full-file rewrites. Copy old_string verbatim "
            "from a prior read_file without the leading '<line_no>\\t' "
            "prefix, including blank lines, indentation, and final newline. "
            "Add surrounding context so old_string is unique. If an edit "
            "fails, retry that file instead of claiming success."
        ),
        "grep": (
            "Search for a literal substring (NOT a regex) across files. "
            "Pass exactly ONE phrase per call. To search for several "
            "alternatives run grep several times. The result lists matching "
            "lines — read it directly instead of opening every matched "
            "file again."
        ),
        "execute": (
            "Run one short shell command. Prefer direct commands/Python "
            "one-liners for benchmark data tasks (CSV, JSON, JSONL, SQLite, "
            "XLSX, logs). Use workspace paths, avoid host paths like "
            "'/Users/...', and match the requested output format exactly."
        ),
    }


def _hybrid_tool_description_overrides() -> dict[str, str]:
    """Tool descriptions for the hybrid benchmark profile."""
    overrides = _tool_description_overrides()
    overrides["grep"] = (
        "Search for a literal substring (NOT a regex) across files. Pass "
        "exactly ONE phrase per call. Results may include path/line prefixes; "
        "when writing final output, strip those prefixes unless the user "
        "explicitly asks for filenames. For counts, count all matches and "
        "write only the number."
    )
    overrides["execute"] = (
        "Run one short shell command. For benchmark data tasks (CSV, JSON, "
        "JSONL, SQLite, XLSX, logs, archives), prefer a compact Python command "
        "or heredoc that writes the requested output file exactly. Do not write "
        "literal shell substitutions like '$(...)' into files. Use workspace "
        "paths, avoid host paths like '/Users/...', and match the requested "
        "output format exactly."
    )
    return overrides


def _build_profile() -> HarnessProfile:
    """Build the selected GigaChat HarnessProfile."""
    selected = _profile_name()
    if selected == "default":
        return HarnessProfile(
            base_system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n",
            tool_description_overrides=_tool_description_overrides(),
            extra_middleware=(ThinkToolMiddleware(),),
        )
    if selected == "hybrid":
        return HarnessProfile(
            base_system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n",
            tool_description_overrides=_hybrid_tool_description_overrides(),
            extra_middleware=(ThinkToolMiddleware(), HybridLoopMiddleware()),
        )
    if selected == "adaptive":
        return HarnessProfile(
            base_system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n",
            tool_description_overrides=_hybrid_tool_description_overrides(),
            extra_middleware=(ThinkToolMiddleware(), AdaptiveLoopMiddleware()),
        )
    if selected == "adaptive-tools":
        return HarnessProfile(
            base_system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n",
            tool_description_overrides=_hybrid_tool_description_overrides(),
            extra_middleware=(ThinkToolMiddleware(), AdaptiveToolRoutingMiddleware()),
        )
    if selected == "pi-lite":
        return HarnessProfile(
            base_system_prompt=f"{PI_LITE_SYSTEM_PROMPT}\n\n",
            tool_description_overrides=_tool_description_overrides(),
            excluded_tools=frozenset({"task", "write_todos"}),
            excluded_middleware=frozenset(
                {
                    "SummarizationMiddleware",
                    "TodoListMiddleware",
                }
            ),
            extra_middleware=(PiLiteLoopMiddleware(),),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
    if selected == "pi-tools":
        return HarnessProfile(
            base_system_prompt=f"{PI_TOOLS_SYSTEM_PROMPT}\n\n",
            tool_description_overrides=_tool_description_overrides(),
            excluded_tools=frozenset(
                {
                    "edit_file",
                    "execute",
                    "glob",
                    "read_file",
                    "task",
                    "write_file",
                    "write_todos",
                }
            ),
            excluded_middleware=frozenset(
                {
                    "SummarizationMiddleware",
                    "TodoListMiddleware",
                }
            ),
            extra_middleware=(PiToolsLoopMiddleware(),),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
    raise ValueError(
        "Unknown DEEPAGENTS_GIGACHAT_PROFILE "
        f"{selected!r}; expected 'default', 'hybrid', 'adaptive', "
        "'adaptive-tools', 'pi-lite', or 'pi-tools'."
    )


def register_harness() -> None:
    """Register the GigaChat HarnessProfile under GigaChat provider keys."""
    profile = _build_profile()

    for provider_key in ("gigachat", "giga"):
        register_harness_profile(provider_key, profile)
