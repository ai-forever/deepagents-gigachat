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
                "surrounding context so old_string is unique within the file."
            ),
            "grep": (
                "Search for a literal substring (NOT a regex) across files. "
                "Pass exactly ONE phrase per call. To search for several "
                "alternatives run grep several times. The result lists matching "
                "lines — read it directly instead of opening every matched "
                "file again."
            ),
            "execute": (
                "Run one short shell command (e.g. 'rm a.txt', 'mv old new', "
                "'mkdir -p logs'). Use this for filesystem operations that the "
                "file tools cannot do (delete, rename, move, chmod). Never "
                "embed multi-line content via sh -c \"...\" or bash -c \"...\" "
                "with double quotes; if you must run a multi-line snippet, use "
                "a single-quoted heredoc (cat <<'EOF' ... EOF). Prefer "
                "write_file / edit_file for changing file content. Use execute "
                "for quick verification of required outputs (e.g., `ls -l`, "
                "`wc -l`) before finishing."
            ),
        },
        extra_middleware=(ThinkToolMiddleware(),),
    )

    for provider_key in ("gigachat", "giga"):
        register_harness_profile(provider_key, profile)
