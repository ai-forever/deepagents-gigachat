"""Run a single task or the whole benchmark against a GigaChat-powered deep agent."""

from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from harness_bench.core import Task, VerifyResult
from harness_bench.tasks import ALL_TASKS, get_task


@dataclass
class TaskRun:
    """The outcome of running a single task."""

    task_id: str
    passed: bool
    message: str
    elapsed_seconds: float
    error: str | None = None
    workspace: Path | None = None


_TRANSIENT_AGENT_ERROR_MARKERS = (
    "authenticationerror: 401",
    "authenticationerror: 403",
    "connecterror",
    "connecttimeout",
    "httpstatuserror: 429",
    "httpstatuserror: 500",
    "httpstatuserror: 502",
    "httpstatuserror: 503",
    "httpstatuserror: 504",
    "readerror",
    "readtimeout",
    "remoteprotocolerror",
    "server disconnected without sending a response",
    "status\":401",
    "status\":403",
    "status\":429",
    "status\":500",
    "status\":502",
    "status\":503",
    "status\":504",
    "temporarily unavailable",
)

_FINALIZATION_RECURSION_LIMIT = 20


def _is_transient_agent_error(error: str | None) -> bool:
    """Return whether an agent exception is worth retrying from scratch."""
    if not error:
        return False
    lowered = error.lower()
    if _is_graph_recursion_error(error):
        return False
    if "during task with name 'model'" in lowered:
        return True
    return any(marker in lowered for marker in _TRANSIENT_AGENT_ERROR_MARKERS)


def _is_graph_recursion_error(error: str | None) -> bool:
    """Return whether an agent exception is a LangGraph recursion-limit stop."""
    return bool(error and "graph_recursion_limit" in error.lower())


def _load_env_from_dotenv() -> None:
    """Best-effort load of .env from the repository root."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Find a .env next to the package — fall back to CWD.
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / ".env",
        repo_root.parent / ".env",
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def build_agent(workspace: Path, *, recursion_limit: int = 80) -> Any:
    """Build a deep agent backed by GigaChat and rooted at `workspace`.

    Imports happen here so `--gold` / `list` modes can run without GigaChat
    credentials configured.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langchain_gigachat import GigaChat

    from deepagents_gigachat import register_harness
    from deepagents_gigachat.pi_tools import build_pi_like_tools

    register_harness()

    backend = LocalShellBackend(
        root_dir=workspace,
        virtual_mode=True,
        inherit_env=True,
    )
    model = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat-3-Ultra"),
        base_url=os.getenv("GIGACHAT_BASE_URL", "https://gigachat.sberdevices.ru/v1"),
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=600,
    )
    tools = None
    if os.getenv("DEEPAGENTS_GIGACHAT_PROFILE", "").strip().lower() == "pi-tools":
        tools = build_pi_like_tools(workspace)
    agent = create_deep_agent(model=model, backend=backend, tools=tools)
    return agent.with_config({"recursion_limit": recursion_limit})


def _ensure_credentials() -> None:
    if os.getenv("GIGACHAT_CREDENTIALS"):
        return
    if os.getenv("GIGACHAT_USER") and os.getenv("GIGACHAT_PASSWORD"):
        return
    raise SystemExit(
        "Не заданы учётные данные GigaChat. "
        "Укажи GIGACHAT_CREDENTIALS либо пару GIGACHAT_USER + GIGACHAT_PASSWORD."
    )


def run_task(
    task: Task,
    *,
    keep_workspace: bool = False,
    recursion_limit: int = 80,
    agent_error_retries: int = 0,
    retry_base_delay: float = 1.0,
    correction_retries: int = 0,
    recursion_recovery_attempts: int = 0,
    finalization_retries: int = 0,
) -> TaskRun:
    """Run a single task end-to-end and return its outcome.

    Args:
        task: The benchmark task to execute.
        keep_workspace: When `True`, the temp workspace directory is not
            deleted after the run — handy for debugging a failure.
        recursion_limit: Cap on agent loop iterations.
        agent_error_retries: Number of fresh-workspace retries for transient
            model/API exceptions. Verifier failures are never retried.
        retry_base_delay: Initial exponential backoff delay in seconds.
        correction_retries: Number of same-workspace corrective attempts after
            verifier failures. Agent exceptions still go through fresh retries.
        recursion_recovery_attempts: Number of same-workspace recovery attempts
            after `GRAPH_RECURSION_LIMIT`. Unlike transient retries, these do
            not reset the workspace.
        finalization_retries: Number of short same-workspace finalization
            attempts after normal correction still leaves a verifier failure.
    """
    attempts = max(agent_error_retries, 0) + 1
    last_run: TaskRun | None = None
    for attempt in range(1, attempts + 1):
        run = _run_task_once(
            task,
            keep_workspace=keep_workspace,
            recursion_limit=recursion_limit,
            correction_retries=correction_retries,
            recursion_recovery_attempts=recursion_recovery_attempts,
            finalization_retries=finalization_retries,
        )
        last_run = run
        if not run.error or not _is_transient_agent_error(run.error):
            return run
        if attempt >= attempts:
            return run
        time.sleep(max(retry_base_delay, 0) * (2 ** (attempt - 1)))

    # The loop always returns, but keep type-checkers happy.
    return last_run or TaskRun(task.id, False, "", 0.0, error="retry loop did not run")


def _run_task_once(
    task: Task,
    *,
    keep_workspace: bool = False,
    recursion_limit: int = 80,
    correction_retries: int = 0,
    recursion_recovery_attempts: int = 0,
    finalization_retries: int = 0,
) -> TaskRun:
    """Run one task attempt in a fresh workspace."""
    workspace_keepalive: TemporaryDirectory | None = None
    try:
        if keep_workspace:
            workspace_path = Path(
                __import__("tempfile").mkdtemp(prefix=f"hb_{task.id}_")
            )
        else:
            workspace_keepalive = TemporaryDirectory(prefix=f"hb_{task.id}_")
            workspace_path = Path(workspace_keepalive.name)

        task.setup(workspace_path)
        started = time.monotonic()
        try:
            agent = build_agent(workspace_path, recursion_limit=recursion_limit)
            agent.invoke({"messages": [{"role": "user", "content": task.prompt}]})
        except Exception:  # noqa: BLE001 — log and surface as failure
            error = traceback.format_exc()
            if _is_graph_recursion_error(error) and recursion_recovery_attempts > 0:
                return _recover_after_recursion_limit(
                    task,
                    workspace_path=workspace_path,
                    started=started,
                    keep_workspace=keep_workspace,
                    original_error=error,
                    attempts=recursion_recovery_attempts,
                    recursion_limit=recursion_limit,
                    correction_retries=correction_retries,
                    finalization_retries=finalization_retries,
                )
            return TaskRun(
                task_id=task.id,
                passed=False,
                message="",
                elapsed_seconds=time.monotonic() - started,
                error=error,
                workspace=workspace_path if keep_workspace else None,
            )
        return _verify_with_corrections(
            task,
            workspace_path=workspace_path,
            agent=agent,
            started=started,
            keep_workspace=keep_workspace,
            correction_retries=correction_retries,
            finalization_retries=finalization_retries,
        )
    finally:
        if workspace_keepalive is not None:
            workspace_keepalive.cleanup()


def _verify_with_corrections(
    task: Task,
    *,
    workspace_path: Path,
    agent: Any,
    started: float,
    keep_workspace: bool,
    correction_retries: int,
    finalization_retries: int,
) -> TaskRun:
    """Verify a workspace and optionally ask the same agent to fix it."""
    result = task.verify(workspace_path)
    for _ in range(max(correction_retries, 0)):
        if result.passed:
            break
        try:
            agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": _correction_prompt(task, result),
                        }
                    ]
                }
            )
        except Exception:  # noqa: BLE001 — log and surface as failure
            return TaskRun(
                task_id=task.id,
                passed=False,
                message=result.message,
                elapsed_seconds=time.monotonic() - started,
                error=traceback.format_exc(),
                workspace=workspace_path if keep_workspace else None,
            )
        result = task.verify(workspace_path)
    if not result.passed and finalization_retries > 0:
        return _finalize_after_verifier_failure(
            task,
            workspace_path=workspace_path,
            started=started,
            keep_workspace=keep_workspace,
            result=result,
            attempts=finalization_retries,
        )
    return TaskRun(
        task_id=task.id,
        passed=result.passed,
        message=result.message,
        elapsed_seconds=time.monotonic() - started,
        workspace=workspace_path if keep_workspace else None,
    )


def _recover_after_recursion_limit(
    task: Task,
    *,
    workspace_path: Path,
    started: float,
    keep_workspace: bool,
    original_error: str,
    attempts: int,
    recursion_limit: int,
    correction_retries: int,
    finalization_retries: int,
) -> TaskRun:
    """Try to finish a partially-mutated workspace after a recursion stop."""
    result = task.verify(workspace_path)
    if result.passed:
        return TaskRun(
            task_id=task.id,
            passed=True,
            message=result.message,
            elapsed_seconds=time.monotonic() - started,
            workspace=workspace_path if keep_workspace else None,
        )

    last_error = original_error
    recovery_limit = min(max(recursion_limit, 1), 20)
    for _ in range(max(attempts, 0)):
        try:
            recovery_agent = build_agent(
                workspace_path,
                recursion_limit=recovery_limit,
            )
            recovery_agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": _recursion_recovery_prompt(
                                task,
                                result,
                                original_error,
                            ),
                        }
                    ]
                }
            )
        except Exception:  # noqa: BLE001 — log and surface as failure
            last_error = f"{original_error}\n\nRecovery error:\n{traceback.format_exc()}"
            break

        run = _verify_with_corrections(
            task,
            workspace_path=workspace_path,
            agent=recovery_agent,
            started=started,
            keep_workspace=keep_workspace,
            correction_retries=correction_retries,
            finalization_retries=finalization_retries,
        )
        if run.passed:
            return run
        result = VerifyResult(False, run.message)
        last_error = original_error

    return TaskRun(
        task_id=task.id,
        passed=False,
        message=result.message,
        elapsed_seconds=time.monotonic() - started,
        error=last_error,
        workspace=workspace_path if keep_workspace else None,
    )


def _finalize_after_verifier_failure(
    task: Task,
    *,
    workspace_path: Path,
    started: float,
    keep_workspace: bool,
    result: VerifyResult,
    attempts: int,
) -> TaskRun:
    """Run a fresh, short, same-workspace pass focused only on final output."""
    last_error: str | None = None
    for _ in range(max(attempts, 0)):
        try:
            finalizer = build_agent(
                workspace_path,
                recursion_limit=_FINALIZATION_RECURSION_LIMIT,
            )
            finalizer.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": _finalization_prompt(task, result),
                        }
                    ]
                }
            )
        except Exception:  # noqa: BLE001 — surface finalizer failures
            last_error = traceback.format_exc()
            break

        result = task.verify(workspace_path)
        if result.passed:
            break

    return TaskRun(
        task_id=task.id,
        passed=result.passed,
        message=result.message,
        elapsed_seconds=time.monotonic() - started,
        error=last_error,
        workspace=workspace_path if keep_workspace else None,
    )


def _correction_prompt(task: Task, result: VerifyResult) -> str:
    """Build a short same-workspace correction request from verifier feedback."""
    detail = result.message.strip() or "verifier failed without details"
    return (
        "The previous attempt did not pass the mechanical verifier.\n"
        "Fix the existing workspace now; do not restart from scratch.\n"
        "Use tools to create or update the requested files, then stop.\n\n"
        f"Original task:\n{task.prompt}\n\n"
        f"Verifier failure:\n{detail}\n\n"
        "Common fixes: create missing output files; if an output file already "
        "exists, overwrite or edit it with `edit_file`/`execute` instead of "
        "`write_file`; remove old paths after renames/moves; strip filename "
        "prefixes from final grep/search output; ignore summary labels like "
        "`total` from `wc -l`; and match exact numeric/string formatting."
    )


def _finalization_prompt(task: Task, result: VerifyResult) -> str:
    """Build a strict final-output repair prompt for a short fresh agent pass."""
    detail = result.message.strip() or "verifier failed without details"
    return (
        "Finalization pass for a benchmark task.\n"
        "The workspace is already set up and may be partially solved. Do not "
        "restart from scratch. You have a small loop budget.\n\n"
        "Your job is only to make the mechanical verifier pass:\n"
        "- Use tools; do not answer in text without changing files if files are wrong.\n"
        "- If an output file is missing, create that exact requested file now.\n"
        "- If an output file exists but content differs, overwrite/edit that exact file.\n"
        "- If the failure says an old path still exists after rename/move, delete the old path.\n"
        "- For grep/search tasks, process every requested file/match, not just the first one.\n"
        "- Strip filenames, line numbers, and summary labels like `total` unless explicitly requested.\n"
        "- Prefer one direct `execute` Python command that writes the final file(s) exactly.\n"
        "- Stop immediately after the minimal successful file-changing action.\n\n"
        f"Original task:\n{task.prompt}\n\n"
        f"Verifier failure to fix:\n{detail}"
    )


def _recursion_recovery_prompt(
    task: Task,
    result: VerifyResult,
    original_error: str,
) -> str:
    """Build a short recovery request for a workspace after graph recursion."""
    detail = result.message.strip() or "verifier failed without details"
    error_tail = original_error.strip().splitlines()[-1]
    return (
        "The previous attempt hit the agent recursion limit before finishing.\n"
        "Continue in the existing workspace; do not restart from scratch.\n"
        "Inspect only what is necessary, then make the minimal file change that "
        "satisfies the task. Prefer one direct tool call or a short Python "
        "script, then stop.\n\n"
        f"Original task:\n{task.prompt}\n\n"
        f"Current verifier failure:\n{detail}\n\n"
        f"Previous agent error:\n{error_tail}"
    )


def run_all(
    task_ids: list[str] | None = None,
    *,
    keep_workspace: bool = False,
    recursion_limit: int = 80,
    concurrency: int = 1,
    agent_error_retries: int = 0,
    retry_base_delay: float = 1.0,
    correction_retries: int = 0,
    recursion_recovery_attempts: int = 0,
    finalization_retries: int = 0,
) -> list[TaskRun]:
    """Run a subset (or all) of the benchmark tasks.

    When `concurrency == 1` (default) tasks run sequentially and progress is
    printed in two lines per task (`→ task_id: name` then `[PASS] ...`).
    When `concurrency > 1` tasks run in a `ThreadPoolExecutor` (each task is
    fully isolated in its own `TemporaryDirectory`, so no synchronization is
    required around the workspace); progress is printed as a single line per
    task in completion order. The returned list is sorted by task id so the
    summary block is deterministic regardless of completion order.
    """
    _load_env_from_dotenv()
    _ensure_credentials()

    targets = [get_task(tid) for tid in task_ids] if task_ids else list(ALL_TASKS)

    if concurrency <= 1:
        results: list[TaskRun] = []
        for task in targets:
            print(f"→ {task.id}: {task.name}")
            run = run_task(
                task,
                keep_workspace=keep_workspace,
                recursion_limit=recursion_limit,
                agent_error_retries=agent_error_retries,
                retry_base_delay=retry_base_delay,
                correction_retries=correction_retries,
                recursion_recovery_attempts=recursion_recovery_attempts,
                finalization_retries=finalization_retries,
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
                run_task,
                task,
                keep_workspace=keep_workspace,
                recursion_limit=recursion_limit,
                agent_error_retries=agent_error_retries,
                retry_base_delay=retry_base_delay,
                correction_retries=correction_retries,
                recursion_recovery_attempts=recursion_recovery_attempts,
                finalization_retries=finalization_retries,
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
                    f"[{completed:3d}/{total}] [{status}] {run.task_id:32s} "
                    f"{run.elapsed_seconds:5.1f}s — {_one_line_detail(run)}"
                )
                if keep_workspace and run.workspace:
                    print(f"           workspace: {run.workspace}")
    results.sort(key=lambda r: _task_sort_key(r.task_id))
    return results


def _task_sort_key(task_id: str) -> tuple[int, str]:
    """Sort task ids by their leading numeric component (`task_03_*` < `task_10_*`)."""
    # Strip the leading "task_" prefix and grab digits up to the next underscore.
    rest = task_id.removeprefix("task_")
    head, _, _ = rest.partition("_")
    try:
        return (int(head), task_id)
    except ValueError:
        return (10**9, task_id)


def _one_line_detail(run: TaskRun) -> str:
    """Squash a `TaskRun`'s message/error into a single informative line.

    For verifier failures the message itself is one line and is used as-is.
    For agent exceptions we surface the last non-empty traceback line — that
    is the actual exception type and message, which is much more useful than
    the leading "Traceback (most recent call last):".
    """
    if run.message:
        first = run.message.splitlines()[0]
        return first
    if run.error:
        lines = [line for line in run.error.splitlines() if line.strip()]
        if not lines:
            return "(unknown error)"
        return lines[-1]
    return "(no detail)"


def summarize(results: list[TaskRun]) -> None:
    """Print a pass/fail summary block at the end of a run."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print()
    print("=" * 64)
    print(f"Passed: {passed}/{total}")
    if passed < total:
        print()
        print("Failures:")
        for r in results:
            if r.passed:
                continue
            print(f"  - {r.task_id}: {_one_line_detail(r)}")


# ---------------------------------------------------------------------------
# Gold sanity check (no LLM): exercises the verifiers themselves.
# ---------------------------------------------------------------------------


def verify_gold(task_ids: list[str] | None = None) -> list[TaskRun]:
    """Apply each task's gold solution to a temp workspace and run the verifier.

    Useful for catching off-by-one bugs in verifier code without spending any
    LLM tokens.
    """
    targets = [get_task(tid) for tid in task_ids] if task_ids else list(ALL_TASKS)

    results: list[TaskRun] = []
    for task in targets:
        with TemporaryDirectory(prefix=f"hb_gold_{task.id}_") as tmp:
            ws = Path(tmp)
            task.setup(ws)
            task.apply_gold(ws)
            start = time.monotonic()
            outcome: VerifyResult = task.verify(ws)
            elapsed = time.monotonic() - start
        run = TaskRun(
            task_id=task.id,
            passed=outcome.passed,
            message=outcome.message,
            elapsed_seconds=elapsed,
        )
        results.append(run)
        status = "OK  " if outcome.passed else "BAD "
        first = outcome.message.splitlines()[0] if outcome.message else ""
        print(f"[{status}] {task.id} ({elapsed * 1000:.1f}ms) — {first}")
    return results
