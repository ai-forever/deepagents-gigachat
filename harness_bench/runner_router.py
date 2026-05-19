"""Route benchmark tasks between DeepAgents and the direct controller."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Literal

from harness_bench.core import Task
from harness_bench.runner import (
    TaskRun,
    _ensure_credentials,
    _load_env_from_dotenv,
    _one_line_detail,
    _task_sort_key,
    run_task,
)
from harness_bench.runner_direct import (
    DEFAULT_ACTION_ERROR_RETRIES,
    DEFAULT_ACTION_TIMEOUT_SECONDS,
    DEFAULT_DIRECT_MODEL,
    DEFAULT_MAX_ACTIONS,
    run_task_direct,
)
from harness_bench.tasks import ALL_TASKS, get_task

DEFAULT_ROUTER_DEEP_PROFILE = "hybrid"
RouterMode = Literal["direct", "deep"]

_DEEP_ROUTE_TAGS = frozenset({"fix", "impl", "pytest", "refactor", "tests"})
_DEEP_EDIT_FORMAT_TAGS = frozenset({"toml"})
_DEEP_PYTHON_EDIT_TAGS = frozenset({"create", "edit"})
_DIRECT_PYTHON_TASK_TAGS = frozenset(
    {
        "compute",
        "convert",
        "csv",
        "execute",
        "grep",
        "json",
        "logs",
        "search",
        "sqlite",
        "xlsx",
        "yaml",
    }
)
_DEEP_PROMPT_MARKERS = (
    "fix the bug",
    "implement ",
    "make pytest",
    "make the tests pass",
    "pytest passes",
    "refactor ",
)
_DIRECT_PROMPT_MARKERS = (
    "move function",
    "перенеси функцию",
)


def route_for_task(task: Task) -> RouterMode:
    """Choose the execution loop from task semantics, not task ids."""
    tags = {tag.lower() for tag in task.tags}
    prompt = task.prompt.lower()
    if "filesystem" in tags:
        return "direct"
    if any(marker in prompt for marker in _DIRECT_PROMPT_MARKERS):
        return "direct"
    if "logs" in tags and "filter" in tags:
        return "deep"
    if {"xlsx", "csv", "json"} <= tags:
        return "deep"
    if tags & _DEEP_ROUTE_TAGS:
        return "deep"
    if "edit" in tags and tags & _DEEP_EDIT_FORMAT_TAGS:
        return "deep"
    if (
        "python" in tags
        and tags & _DEEP_PYTHON_EDIT_TAGS
        and not tags & _DIRECT_PYTHON_TASK_TAGS
    ):
        return "deep"

    if ".toml" in prompt or "pyproject.toml" in prompt:
        return "deep"
    if any(marker in prompt for marker in _DEEP_PROMPT_MARKERS):
        return "deep"

    return "direct"


def run_task_router(
    task: Task,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    deep_profile: str | None = DEFAULT_ROUTER_DEEP_PROFILE,
    keep_workspace: bool = False,
    recursion_limit: int = 80,
    agent_error_retries: int = 0,
    retry_base_delay: float = 1.0,
    action_timeout: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    action_error_retries: int = DEFAULT_ACTION_ERROR_RETRIES,
) -> TaskRun:
    """Run one task through the routed benchmark loop."""
    mode = route_for_task(task)
    if mode == "direct":
        return run_task_direct(
            task,
            model_name=model_name,
            keep_workspace=keep_workspace,
            action_timeout=action_timeout,
            max_actions=max_actions,
            action_error_retries=action_error_retries,
        )

    env = _deep_agent_env(model_name=model_name, deep_profile=deep_profile)
    with _temporary_env(env):
        return run_task(
            task,
            keep_workspace=keep_workspace,
            recursion_limit=recursion_limit,
            agent_error_retries=agent_error_retries,
            retry_base_delay=retry_base_delay,
            correction_retries=0,
            recursion_recovery_attempts=0,
            finalization_retries=0,
        )


def run_all_router(
    task_ids: list[str] | None = None,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    deep_profile: str = DEFAULT_ROUTER_DEEP_PROFILE,
    keep_workspace: bool = False,
    recursion_limit: int = 80,
    agent_error_retries: int = 0,
    retry_base_delay: float = 1.0,
    action_timeout: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    action_error_retries: int = DEFAULT_ACTION_ERROR_RETRIES,
    concurrency: int = 1,
) -> list[TaskRun]:
    """Run tasks through the semantic router.

    The router intentionally avoids verifier feedback. DeepAgents receives the
    same initial task prompt that `run` would use; direct tasks keep the bounded
    missing-artifact loop from `run-direct`.
    """
    _load_env_from_dotenv()
    _ensure_credentials()

    targets = [get_task(tid) for tid in task_ids] if task_ids else list(ALL_TASKS)
    env = _deep_agent_env(model_name=model_name, deep_profile=deep_profile)
    with _temporary_env(env):
        if concurrency <= 1:
            return _run_all_router_sequential(
                targets,
                model_name=model_name,
                keep_workspace=keep_workspace,
                recursion_limit=recursion_limit,
                agent_error_retries=agent_error_retries,
                retry_base_delay=retry_base_delay,
                action_timeout=action_timeout,
                max_actions=max_actions,
                action_error_retries=action_error_retries,
            )

        return _run_all_router_concurrent(
            targets,
            model_name=model_name,
            keep_workspace=keep_workspace,
            recursion_limit=recursion_limit,
            agent_error_retries=agent_error_retries,
            retry_base_delay=retry_base_delay,
            action_timeout=action_timeout,
            max_actions=max_actions,
            action_error_retries=action_error_retries,
            concurrency=concurrency,
        )


def _run_all_router_sequential(
    tasks: list[Task],
    *,
    model_name: str,
    keep_workspace: bool,
    recursion_limit: int,
    agent_error_retries: int,
    retry_base_delay: float,
    action_timeout: int,
    max_actions: int,
    action_error_retries: int,
) -> list[TaskRun]:
    results: list[TaskRun] = []
    for task in tasks:
        mode = route_for_task(task)
        print(f"-> {task.id}: {task.name} [{_mode_label(mode)}]")
        run = run_task_router(
            task,
            model_name=model_name,
            deep_profile=None,
            keep_workspace=keep_workspace,
            recursion_limit=recursion_limit,
            agent_error_retries=agent_error_retries,
            retry_base_delay=retry_base_delay,
            action_timeout=action_timeout,
            max_actions=max_actions,
            action_error_retries=action_error_retries,
        )
        results.append(run)
        status = "PASS" if run.passed else "FAIL"
        print(f"  [{status}] {run.elapsed_seconds:5.1f}s - {_one_line_detail(run)}")
        if keep_workspace and run.workspace:
            print(f"  workspace: {run.workspace}")
    return results


def _run_all_router_concurrent(
    tasks: list[Task],
    *,
    model_name: str,
    keep_workspace: bool,
    recursion_limit: int,
    agent_error_retries: int,
    retry_base_delay: float,
    action_timeout: int,
    max_actions: int,
    action_error_retries: int,
    concurrency: int,
) -> list[TaskRun]:
    print_lock = threading.Lock()
    completed = 0
    total = len(tasks)
    results: list[TaskRun] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                run_task_router,
                task,
                model_name=model_name,
                deep_profile=None,
                keep_workspace=keep_workspace,
                recursion_limit=recursion_limit,
                agent_error_retries=agent_error_retries,
                retry_base_delay=retry_base_delay,
                action_timeout=action_timeout,
                max_actions=max_actions,
                action_error_retries=action_error_retries,
            ): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            run = future.result()
            results.append(run)
            with print_lock:
                completed += 1
                status = "PASS" if run.passed else "FAIL"
                print(
                    f"[{completed:3d}/{total}] [{status}] {run.task_id:32s} "
                    f"{_mode_label(route_for_task(task)):13s} "
                    f"{run.elapsed_seconds:5.1f}s - {_one_line_detail(run)}"
                )
                if keep_workspace and run.workspace:
                    print(f"           workspace: {run.workspace}")
    results.sort(key=lambda r: _task_sort_key(r.task_id))
    return results


def _deep_agent_env(
    *,
    model_name: str,
    deep_profile: str | None,
) -> dict[str, str | None]:
    env: dict[str, str | None] = {"GIGACHAT_MODEL": model_name}
    if deep_profile is not None:
        env["DEEPAGENTS_GIGACHAT_PROFILE"] = deep_profile
    return env


@contextmanager
def _temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _mode_label(mode: RouterMode) -> str:
    if mode == "deep":
        return "deep/hybrid"
    return "direct"
