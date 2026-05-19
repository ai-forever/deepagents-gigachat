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
import random
import shlex
import shutil
import subprocess
import threading
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from harness_bench.core import Task
from harness_bench.runner import (
    TaskRun,
    _load_env_from_dotenv,
    _one_line_detail,
    _task_sort_key,
    summarize,
)
from harness_bench.tasks import ALL_TASKS, get_task

DEFAULT_CLI_COMMAND = (
    "free-code -p --model haiku --dangerously-skip-permissions"
)
"""Default CLI invocation. The prompt is appended as the final argument."""

DEFAULT_PI_COMMAND = (
    "pi -p --no-session --no-context-files --no-extensions --no-skills "
    "--no-prompt-templates --no-themes --tools read,bash,edit,write,grep,find,ls"
)
"""Baseline `pi` invocation used by the dedicated `run-pi` command."""

DEFAULT_PI_ENV = {
    "PI_SKIP_VERSION_CHECK": "1",
    "PI_TELEMETRY": "0",
}
"""Environment tweaks for quieter, more reproducible `pi` benchmark runs."""

DEFAULT_TIMEOUT_SECONDS = 600
"""Per-task timeout in seconds. Some tasks need pytest + multiple file edits."""

GIGACHAT_TRANSIENT_403 = "GigaChat streaming request failed with status 403"
"""Known intermittent GigaChat gateway failure worth retrying in benchmark runs."""


def build_pi_cli_command(
    *,
    model_name: str | None = None,
    provider: str | None = None,
    thinking: str | None = None,
    extensions: Sequence[str] | None = None,
) -> str:
    """Build the default `pi` CLI invocation with optional model overrides."""
    argv = shlex.split(DEFAULT_PI_COMMAND)
    for extension in extensions or ():
        argv.extend(["-e", extension])
    if provider:
        argv.extend(["--provider", provider])
    if model_name:
        argv.extend(["--model", model_name])
    if thinking:
        argv.extend(["--thinking", thinking])
    return shlex.join(argv)


def run_task_cli(
    task: Task,
    *,
    cli_command: str = DEFAULT_CLI_COMMAND,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    keep_workspace: bool = False,
    env_overrides: dict[str, str] | None = None,
    transient_403_retries: int = 0,
    retry_base_delay: float = 2.0,
) -> TaskRun:
    """Run a single task via the CLI agent and return its `TaskRun` result."""
    workspace_keepalive: TemporaryDirectory | None = None
    try:
        if keep_workspace:
            workspace_path = Path(mkdtemp(prefix=f"hb_cli_{task.id}_"))
        else:
            workspace_keepalive = TemporaryDirectory(prefix=f"hb_cli_{task.id}_")
            workspace_path = Path(workspace_keepalive.name)

        started = time.monotonic()
        attempts = max(0, transient_403_retries) + 1
        last_run: TaskRun | None = None

        for attempt in range(1, attempts + 1):
            _reset_workspace(workspace_path)
            task.setup(workspace_path)

            argv = shlex.split(cli_command) + [task.prompt]
            env = os.environ.copy()
            if env_overrides:
                env.update(env_overrides)
            try:
                result = subprocess.run(  # noqa: S603 — trusted local benchmark
                    argv,
                    cwd=workspace_path,
                    env=env,
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
            if attempt > 1:
                message = f"{message} (attempt {attempt}/{attempts})"

            last_run = TaskRun(
                task_id=task.id,
                passed=outcome.passed,
                message=message,
                elapsed_seconds=time.monotonic() - started,
                workspace=workspace_path if keep_workspace else None,
            )

            if (
                not last_run.passed
                and attempt < attempts
                and _is_retryable_gigachat_403(result)
            ):
                _sleep_before_retry(attempt, retry_base_delay)
                continue
            return last_run

        # The loop always returns, but keep mypy and future edits honest.
        assert last_run is not None
        return last_run
    finally:
        if workspace_keepalive is not None:
            workspace_keepalive.cleanup()


def _reset_workspace(workspace_path: Path) -> None:
    """Reset a benchmark temp workspace before each retry attempt."""
    workspace_path.mkdir(parents=True, exist_ok=True)
    for child in workspace_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _is_retryable_gigachat_403(result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode == 0:
        return False
    output = f"{result.stderr}\n{result.stdout}"
    return GIGACHAT_TRANSIENT_403 in output


def _sleep_before_retry(attempt: int, base_delay: float) -> None:
    delay = max(0.0, base_delay) * (4 ** (attempt - 1))
    jitter = random.uniform(0.0, min(1.0, delay * 0.25)) if delay else 0.0
    time.sleep(delay + jitter)


def run_all_cli(
    task_ids: list[str] | None = None,
    *,
    cli_command: str = DEFAULT_CLI_COMMAND,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    keep_workspace: bool = False,
    concurrency: int = 1,
    env_overrides: dict[str, str] | None = None,
    transient_403_retries: int = 0,
    retry_base_delay: float = 2.0,
) -> list[TaskRun]:
    """Run a subset (or all) of the benchmark via the CLI agent."""
    _load_env_from_dotenv()
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
                env_overrides=env_overrides,
                transient_403_retries=transient_403_retries,
                retry_base_delay=retry_base_delay,
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
                env_overrides=env_overrides,
                transient_403_retries=transient_403_retries,
                retry_base_delay=retry_base_delay,
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
    "DEFAULT_PI_COMMAND",
    "DEFAULT_PI_ENV",
    "DEFAULT_TIMEOUT_SECONDS",
    "build_pi_cli_command",
    "run_all_cli",
    "run_task_cli",
    "summarize",
]

# Keep `os` imported in case future versions need to inspect env / PATH for
# locating the CLI binary or setting per-task env vars.
_ = os
