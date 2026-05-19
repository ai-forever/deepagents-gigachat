"""Small pi-like tool facade for benchmark experiments."""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Annotated

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class EditItem(BaseModel):
    """One exact replacement against the original file content."""

    oldText: str = Field(description="Exact text to replace.")
    newText: str = Field(description="Replacement text.")


class EditInput(BaseModel):
    """Input schema matching pi's batch edit shape."""

    path: str = Field(description="Path to the file to edit.")
    edits: list[EditItem] = Field(
        description=(
            "One or more exact replacements. Each oldText is matched against "
            "the original file, not incrementally."
        ),
    )


def build_pi_like_tools(workspace: Path) -> list[BaseTool]:
    """Create a minimal pi-like toolset rooted at a benchmark workspace."""

    root = workspace.resolve()

    def resolve_workspace_path(path: str) -> Path:
        candidate = path.strip() or "."
        if candidate.startswith("/"):
            candidate = candidate.lstrip("/")
        resolved = (root / candidate).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"path escapes workspace: {path!r}")
        return resolved

    def display_path(path: Path) -> str:
        rel = path.resolve().relative_to(root)
        return str(rel).replace("\\", "/") or "."

    def error(exc: BaseException) -> str:
        return f"Error: {exc}"

    def read(
        path: Annotated[str, "Path to the file to read."],
        offset: Annotated[int | None, "1-indexed first line to read."] = None,
        limit: Annotated[int | None, "Maximum number of lines to return."] = None,
    ) -> str:
        """Read a text file from the workspace."""
        try:
            target = resolve_workspace_path(path)
            text = target.read_text(encoding="utf-8")
            if offset is None and limit is None:
                return text
            lines = text.splitlines(keepends=True)
            start = max((offset or 1) - 1, 0)
            end = None if limit is None else start + max(limit, 0)
            return "".join(lines[start:end])
        except Exception as exc:  # noqa: BLE001 - surface as tool observation
            return error(exc)

    def write(
        path: Annotated[str, "Path to create or overwrite."],
        content: Annotated[str, "Complete file content."],
    ) -> str:
        """Create or overwrite a text file."""
        try:
            target = resolve_workspace_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} bytes to {display_path(target)}"
        except Exception as exc:  # noqa: BLE001 - surface as tool observation
            return error(exc)

    def edit_impl(path: str, edits: list[EditItem]) -> str:
        try:
            target = resolve_workspace_path(path)
            original = target.read_text(encoding="utf-8")
            spans: list[tuple[int, int, str]] = []
            for item in edits:
                count = original.count(item.oldText)
                if count != 1:
                    raise ValueError(
                        f"oldText must match exactly once in {path!r}, found {count}"
                    )
                start = original.index(item.oldText)
                end = start + len(item.oldText)
                spans.append((start, end, item.newText))
            spans.sort()
            for (_, prev_end, _), (next_start, _, _) in zip(
                spans,
                spans[1:],
                strict=False,
            ):
                if next_start < prev_end:
                    raise ValueError("edits must not overlap")
            chunks: list[str] = []
            cursor = 0
            for start, end, new_text in spans:
                chunks.append(original[cursor:start])
                chunks.append(new_text)
                cursor = end
            chunks.append(original[cursor:])
            updated = "".join(chunks)
            target.write_text(updated, encoding="utf-8")
            return f"replaced {len(edits)} block(s) in {display_path(target)}"
        except Exception as exc:  # noqa: BLE001 - surface as tool observation
            return error(exc)

    def bash(
        command: Annotated[str, "Shell command to execute in the workspace."],
        timeout: Annotated[int | None, "Timeout in seconds."] = 120,
    ) -> str:
        """Run a shell command in the workspace."""
        try:
            effective_timeout = 120 if timeout is None else min(max(timeout, 1), 600)
            try:
                result = subprocess.run(  # noqa: S602,S603 - trusted local benchmark
                    command,
                    cwd=root,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=effective_timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                output = "".join(
                    part or ""
                    for part in (
                        exc.stdout if isinstance(exc.stdout, str) else "",
                        exc.stderr if isinstance(exc.stderr, str) else "",
                    )
                )
                return f"{output}\nCommand timed out after {effective_timeout}s".strip()
            output = "".join(part for part in (result.stdout, result.stderr) if part)
            status = f"exit={result.returncode}"
            return f"{output}\n{status}".strip() if output else status
        except Exception as exc:  # noqa: BLE001 - surface as tool observation
            return error(exc)

    def find(
        pattern: Annotated[str, "Filename glob, e.g. '*.py' or '**/*.csv'"] = "*",
        path: Annotated[str, "Directory to search from."] = ".",
    ) -> str:
        """Find files by glob pattern."""
        try:
            base = resolve_workspace_path(path)
            if not base.exists():
                return ""
            matches = [
                display_path(p)
                for p in base.rglob("*")
                if p.is_file() and fnmatch.fnmatch(display_path(p), pattern)
            ]
            return "\n".join(sorted(matches))
        except Exception as exc:  # noqa: BLE001 - surface as tool observation
            return error(exc)

    edit_tool = StructuredTool.from_function(
        name="edit",
        description=(
            "Edit a single file using one or more exact replacements. "
            "Input shape: path plus edits=[{oldText,newText}, ...]."
        ),
        func=edit_impl,
        args_schema=EditInput,
    )
    return [
        StructuredTool.from_function(read),
        StructuredTool.from_function(write),
        edit_tool,
        StructuredTool.from_function(bash),
        StructuredTool.from_function(find),
    ]
