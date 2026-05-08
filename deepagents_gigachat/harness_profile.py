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
    """Register the GigaChat HarnessProfile under GigaChat provider keys."""
    profile = HarnessProfile(
        base_system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n",
        tool_description_overrides={
            "write_file": (
                "Create or overwrite a file. Use relative paths like 'foo.py' or "
                "'src/foo.py'. Do NOT start file_path with '/'. "
                "Use write_file for new files or full rewrites; use edit_file for small "
                "changes. Do not include line-number prefixes from read_file output."
            ),
            "edit_file": (
                "Edit a file by replacing one exact old_string with new_string. "
                "Always read_file first and copy old_string 1:1 from the read output "
                "(no line-number prefixes), including blank lines and indentation. "
                "Include 3-8 lines of context so old_string is unique."
            ),
            "grep": (
                "Search for a literal text pattern. Pattern is NOT regex. "
                "Pass exactly ONE literal phrase per call. "
                "For OR behavior run multiple grep calls."
            ),
            "execute": (
                "Run a shell command. Never embed multiline content inside "
                "sh -c \"...\" or bash -c \"...\"; use a single-quoted heredoc "
                "(cat <<'EOF' ... EOF) or pipe via stdin. Prefer write_file/edit_file "
                "for file content changes."
            ),
        },
        extra_middleware=(ThinkToolMiddleware(),),
    )

    for provider_key in ("gigachat", "giga"):
        register_harness_profile(provider_key, profile)
