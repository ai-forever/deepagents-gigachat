"""Drive each benchmark task through an external CLI agent.

The default GigaChat runner builds an in-process `deepagents` agent. This
module is the alternative: for each task we shell out to a CLI agent (e.g.
`free-code` / Claude Code CLI) inside a fresh temp workspace, then run the
same verifier against the resulting files. That gives us an apples-to-apples
score for "what fraction of the bench would this CLI solve" without changing
the task set.

The CLI command is configurable, defaulting to:

    free-code -p --model haiku --dangerously-skip-permissions <prompt>

The prompt is passed as the last positional argument. We always set `cwd` to
the per-task temp directory and `--add-dir` is not needed because the CLI
defaults to operating on its own cwd.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from harness_bench.core import Task
from harness_bench.runner import TaskRun, _one_line_detail, _task_sort_key, summarize
from harness_bench.tasks import ALL_TASKS, get_task

DEFAULT_CLI_COMMAND = (
    "free-code -p --model haiku --dangerously-skip-permissions"
)
"""Default CLI invocation. The prompt is appended as the final argument."""

DEFAULT_TIMEOUT_SECONDS = 600
"""Per-task timeout in seconds. Some tasks need pytest + multiple file edits."""


def run_task_cli(
    task: Task,
    *,
    cli_command: str = DEFAULT_CLI_COMMAND,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    keep_workspace: bool = False,
) -> TaskRun:
    """Run a single task via the CLI agent and return its `TaskRun` result."""
    workspace_keepalive: TemporaryDirectory | None = None
    try:
        if keep_workspace:
            workspace_path = Path(mkdtemp(prefix=f"hb_cli_{task.id}_"))
        else:
            workspace_keepalive = TemporaryDirectory(prefix=f"hb_cli_{task.id}_")
            workspace_path = Path(workspace_keepalive.name)

        task.setup(workspace_path)
        started = time.monotonic()

        argv = shlex.split(cli_command) + [task.prompt]
        try:
            result = subprocess.run(  # noqa: S603 — trusted local benchmark
                argv,
                cwd=workspace_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return TaskRun(
                task_id=task.id,
                passed=False,
                message="",
                elapsed_seconds=time.monotonic() - started,
                error=f"CLI timed out after {timeout}s",
                workspace=workspace_path if keep_workspace else None,
            )
        except FileNotFoundError as exc:
            return TaskRun(
                task_id=task.id,
                passed=False,
                message="",
                elapsed_seconds=time.monotonic() - started,
                error=f"CLI executable not found: {exc}",
                workspace=workspace_path if keep_workspace else None,
            )
        except Exception:  # noqa: BLE001 — surface as failure
            return TaskRun(
                task_id=task.id,
                passed=False,
                message="",
                elapsed_seconds=time.monotonic() - started,
                error=traceback.format_exc(),
                workspace=workspace_path if keep_workspace else None,
            )

        # We don't fail the task on non-zero CLI exit — the CLI sometimes exits
        # non-zero while still doing useful work. Trust the verifier.
        outcome = task.verify(workspace_path)
        message = outcome.message
        if not outcome.passed and result.returncode != 0:
            tail = (result.stderr or result.stdout).strip()[-300:]
            message = f"{outcome.message} | CLI exit={result.returncode}: {tail!r}"
        return TaskRun(
            task_id=task.id,
            passed=outcome.passed,
            message=message,
            elapsed_seconds=time.monotonic() - started,
            workspace=workspace_path if keep_workspace else None,
        )
    finally:
        if workspace_keepalive is not None:
            workspace_keepalive.cleanup()


def run_all_cli(
    task_ids: list[str] | None = None,
    *,
    cli_command: str = DEFAULT_CLI_COMMAND,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    keep_workspace: bool = False,
    concurrency: int = 1,
) -> list[TaskRun]:
    """Run a subset (or all) of the benchmark via the CLI agent."""
    targets = [get_task(tid) for tid in task_ids] if task_ids else list(ALL_TASKS)

    if concurrency <= 1:
        results: list[TaskRun] = []
        for task in targets:
            print(f"→ {task.id}: {task.name}")
            run = run_task_cli(
                task,
                cli_command=cli_command,
                timeout=timeout,
                keep_workspace=keep_workspace,
            )
            results.append(run)
            status = "PASS" if run.passed else "FAIL"
            print(f"  [{status}] {run.elapsed_seconds:5.1f}s — {_one_line_detail(run)}")
            if keep_workspace and run.workspace:
                print(f"  workspace: {run.workspace}")
        return results

    print_lock = threading.Lock()
    completed = 0
    total = len(targets)
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                run_task_cli,
                task,
                cli_command=cli_command,
                timeout=timeout,
                keep_workspace=keep_workspace,
            ): task
            for task in targets
        }
        for future in as_completed(future_to_task):
            run = future.result()
            results.append(run)
            with print_lock:
                completed += 1
                status = "PASS" if run.passed else "FAIL"
                print(
                    f"[{completed:3d}/{total}] [{status}] {run.task_id:36s} "
                    f"{run.elapsed_seconds:5.1f}s — {_one_line_detail(run)}"
                )
                if keep_workspace and run.workspace:
                    print(f"           workspace: {run.workspace}")
    results.sort(key=lambda r: _task_sort_key(r.task_id))
    return results


__all__ = [
    "DEFAULT_CLI_COMMAND",
    "DEFAULT_TIMEOUT_SECONDS",
    "run_all_cli",
    "run_task_cli",
    "summarize",
]

# Keep `os` imported in case future versions need to inspect env / PATH for
# locating the CLI binary or setting per-task env vars.
_ = os
