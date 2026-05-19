"""Example: run the public routed workspace runtime on a temporary project."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from deepagents_gigachat import invoke_routed


def main() -> None:
    with TemporaryDirectory(prefix="deepagents_gigachat_") as tmp_dir:
        workspace = Path(tmp_dir)
        (workspace / "numbers.csv").write_text("value\n2\n5\n8\n", encoding="utf-8")

        result = invoke_routed(
            "Read numbers.csv, sum the value column, and write the integer total to total.txt.",
            workspace=workspace,
        )

        print("=" * 60)
        print("Workspace:", workspace)
        print("Execution route:", result.decision.execution_route)
        print("Tool route:", result.decision.tool_route)
        print("total.txt:", (workspace / "total.txt").read_text(encoding="utf-8").strip())
        print("=" * 60)


if __name__ == "__main__":
    main()
