"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from deepagents import HarnessProfile, register_harness_profile
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import Field

from deepagents_gigachat.prompts import BASE_SYSTEM_PROMPT


@tool("think")
def _think(thought: str = Field(..., description="A thought to think about.")) -> str:
    """Use this tool as scratchpad to structure intermediate reasoning."""
    return thought


class ThinkToolMiddleware(AgentMiddleware):
    """Inject the local `think` tool into the default toolset."""

    tools = [_think]


class VirtualFilesystemRootMiddleware(AgentMiddleware):
    """Keep virtual absolute paths anchored to the project filesystem root."""

    _filesystem_tool_names = frozenset(
        {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}
    )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        self._enable_virtual_mode_for_filesystem_tool(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        self._enable_virtual_mode_for_filesystem_tool(request)
        return await handler(request)

    @classmethod
    def _enable_virtual_mode_for_filesystem_tool(cls, request: ToolCallRequest) -> None:
        if request.tool_call.get("name") not in cls._filesystem_tool_names:
            return

        filesystem_middleware = cls._filesystem_middleware_from_tool(request.tool)
        if filesystem_middleware is None:
            return

        cls._enable_virtual_mode(getattr(filesystem_middleware, "backend", None))

    @classmethod
    def _filesystem_middleware_from_tool(cls, tool_object: Any) -> Any | None:
        tool_func = getattr(tool_object, "func", None)
        closure = getattr(tool_func, "__closure__", None)
        if not closure:
            return None

        for cell in closure:
            try:
                candidate = cell.cell_contents
            except ValueError:
                continue
            if isinstance(candidate, FilesystemMiddleware):
                return candidate

        return None

    @classmethod
    def _enable_virtual_mode(cls, backend: Any) -> None:
        if backend is None:
            return

        if hasattr(backend, "virtual_mode"):
            backend.virtual_mode = True

        routes = getattr(backend, "routes", None)
        if isinstance(routes, dict):
            for routed_backend in routes.values():
                cls._enable_virtual_mode(routed_backend)


def register_harness() -> None:
    """Register the GigaChat HarnessProfile under GigaChat provider keys.

    Called automatically by `deepagents` lazy bootstrap when the package
    is installed (via the `deepagents.harness_profiles` entry point).
    Safe to call directly as well — `register_harness_profile` merges
    on top of any existing registration.
    """
    profile = HarnessProfile(
        base_system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n",
        tool_description_overrides={
            "ls": (
                "Lists all files in a directory. "
                "Use absolute virtual paths rooted at the current project working directory. "
                "Do NOT use host OS paths like '/Users/...'."
            ),
            "read_file": (
                "Reads a file from the filesystem. "
                "The file_path must be an absolute virtual path starting with '/'. "
                "Use offset/limit for large files (>200 lines) — pass integers, not strings. "
                "Always read a file before editing it. "
                "If a file is not found at the expected path, use glob to locate it by basename."
            ),
            "write_file": (
                "Create or overwrite a file with exact content. Use absolute virtual paths starting with '/'. "
                "A path like '/game_of_life.py' means a file in the current project root, not the host OS filesystem root. "
                "Never use host OS paths like '/Users/...'. "
                "Use write_file for new files or full rewrites; for small changes to existing files, use edit_file instead. "
                "Ensure the parent directory exists before writing. "
                "Do NOT include line-number prefixes from read_file output when writing content. "
                "End the file with a trailing newline."
            ),
            "glob": (
                "List files matching a pattern. Use absolute patterns like '/**/*.py'. "
                "Useful for discovering file locations by basename before editing."
            ),
            "grep": (
                "Search for a text pattern across files. "
                "Pattern matching is literal text, NOT regex. "
                "HARD RULE: pattern must represent ONE literal phrase only. "
                "Never put multiple alternatives into one pattern. "
                "DO NOT use '|', '||', regex groups, or regex syntax for OR. "
                "If pattern contains '|', it is interpreted literally and almost always fails. "
                "WRONG: pattern='os.getenv|load_dotenv|env|config|settings|getenv'. "
                "RIGHT: run multiple grep calls: "
                "1) pattern='os.getenv', "
                "2) pattern='load_dotenv', "
                "3) pattern='getenv', "
                "4) pattern='config', "
                "5) pattern='settings'. "
                "Before each grep call, quickly verify: "
                "(a) single literal phrase, "
                "(b) no pipe '|', "
                "(c) optional path/glob uses absolute virtual paths. "
                "Special characters like '|', '(', ')', '[', ']' are treated as plain characters, not operators."
            ),
            "edit_file": (
                "Edit an existing file by replacing one exact old_string with new_string. "
                "Before edit_file always read the target file with read_file. "
                "Use absolute virtual paths, not '/Users/...'. "
                "Copy old_string 1:1 from read_file output (without line numbers), "
                "including blank lines and indentation. Include 3-8 lines of context "
                "above and below so old_string is unique. If edit_file returns "
                "'String not found', do not guess: re-read the file with more context and "
                "rebuild old_string from fresh content. "
                "WRONG: old_string contains line-number prefixes like '40\\tdef foo():'. "
                "RIGHT: remove prefixes and use exact code text 'def foo():'. "
                "WRONG: old_string is a hand-written approximation or misses blank lines/indentation. "
                "RIGHT: copy exact bytes from read_file, preserving spacing. "
                "WRONG: replacing a common short fragment like 'return kwargs' without unique context. "
                "RIGHT: include surrounding lines (function signature + nearby lines) so the match is unique. "
                "After each successful edit, re-read the file to confirm the change. "
                "If new_string uses a name that needs import, ensure the import exists."
            ),
            "execute": (
                "Execute a shell command. Use this for running tests, builds, package commands, "
                "and short scripts. Before commands that create files or directories, verify the "
                "parent directory with ls. Quote file paths containing spaces with double quotes. "
                "CRITICAL shell-quoting rule: never put multiline content inside sh -c \"...\", "
                "bash -c \"...\", or tools.py write \"...\". Newlines and quotes inside those "
                "strings often break the shell with errors like 'unexpected EOF while looking for "
                "matching quote'. For multiline content, use a single-quoted heredoc such as "
                "cat > /path/file <<'EOF' ... EOF, or pipe the content through stdin. "
                "Prefer write_file/edit_file for file content changes; use execute only when a "
                "shell command is actually needed. Avoid find/grep/cat/head/tail; use glob, grep, "
                "and read_file tools instead."
            ),
        },
        extra_middleware=(ThinkToolMiddleware(), VirtualFilesystemRootMiddleware()),
    )

    for provider_key in ("gigachat", "giga"):
        register_harness_profile(provider_key, profile)
