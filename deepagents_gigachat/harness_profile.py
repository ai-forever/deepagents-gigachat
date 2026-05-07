"""HarnessProfile setup for GigaChat."""

from __future__ import annotations

from deepagents import HarnessProfile, register_harness_profile
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import tool
from pydantic import Field

from deepagents_gigachat.prompts import BASE_SYSTEM_PROMPT


@tool("think")
def _think(thought: str = Field(..., description="A thought to think about.")) -> str:
    """Use this tool as scratchpad to structure intermediate reasoning."""
    return thought


class ThinkToolMiddleware(AgentMiddleware):
    """Inject the local `think` tool into the default toolset."""

    tools = [_think]


def register_harness() -> None:
    """Register the GigaChat HarnessProfile under the `giga` provider key.

    Called automatically by `deepagents` lazy bootstrap when the package
    is installed (via the `deepagents.harness_profiles` entry point).
    Safe to call directly as well — `register_harness_profile` merges
    on top of any existing registration.
    """
    register_harness_profile(
        "giga",
        HarnessProfile(
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
            },
            extra_middleware=(ThinkToolMiddleware(),),
        ),
    )
