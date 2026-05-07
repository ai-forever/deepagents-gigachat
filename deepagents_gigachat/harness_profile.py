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
                    "Use absolute virtual paths rooted at the current project working directory, "
                    "for example '/gigachat_profiles.py' or '/tools'. "
                    "Do NOT use host OS paths like '/Users/...'."
                ),
                "read_file": (
                    "Reads a file from the filesystem. "
                    "The file_path must be an absolute virtual path starting with '/'. "
                    "In this environment, '/...' is relative to the project root (current working directory), "
                    "so use '/gigachat_profiles.py' instead of '/Users/...'. "
                    "Use offset/limit for large files, and always read a file before editing it."
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
                    "Special characters like '|', '(', ')', '[', ']' are treated as plain characters, not operators. "
                    "Paths and optional search roots should use absolute virtual paths (starting with '/')."
                ),
                "edit_file": (
                    "Edit an existing file by replacing one exact old_string with new_string. "
                    "Before edit_file always read the target file with read_file. "
                    "Use absolute virtual paths (for example '/gigachat_profiles.py'), not '/Users/...'. "
                    "Copy old_string 1:1 from read_file output (without line numbers), "
                    "including blank lines and indentation. Include 3-8 lines of context "
                    "above and below so old_string is unique. If edit_file returns "
                    "'String not found', do not guess: re-read with more context and "
                    "rebuild old_string. "
                    "WRONG: old_string contains line-number prefixes like '40\\tdef foo():' from read_file output. "
                    "RIGHT: remove prefixes and use exact code text 'def foo():'. "
                    "WRONG: old_string is a hand-written approximation or misses blank lines/indentation. "
                    "RIGHT: copy exact bytes from read_file, preserving spacing. "
                    "WRONG: replacing a common short fragment like 'return kwargs' without unique context. "
                    "RIGHT: include surrounding lines (function signature + nearby lines) so the match is unique. "
                    "After each successful edit_file call, run a quick import sanity check on the same file: "
                    "if new_string introduces or keeps a name that needs import (for example 'os', 'Path'), "
                    "ensure the import exists; if missing, add it immediately before finishing."
                ),
            },
            extra_middleware=(ThinkToolMiddleware(),),
        ),
    )
