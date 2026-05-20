"""Route benchmark tasks between DeepAgents and the direct controller."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from deepagents_gigachat.orchestrator import (
    build_deep_agent_env as _deep_agent_env,
)
from deepagents_gigachat.orchestrator import (
    run_routed,
)
from deepagents_gigachat.orchestrator import (
    temporary_env as _temporary_env,
)
from deepagents_gigachat.routing import (
    ExecutionRoute as RouterMode,
)
from deepagents_gigachat.routing import (
    RoutingDecision,
    RoutingStrategy,
    build_routing_input,
    route_task,
    route_task_with_model,
)
from deepagents_gigachat.runtime import build_model as _build_router_model
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


def _routing_input_for_task(task: Task, *, use_routing_hints: bool = True):
    hints = task.tags if use_routing_hints else ()
    return build_routing_input(task.prompt, hints=hints)


def _route_decision_for_task(
    task: Task,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    router_mode: RoutingStrategy = "rules",
    router_model_name: str | None = None,
    use_routing_hints: bool = True,
) -> RoutingDecision:
    routing_input = _routing_input_for_task(task, use_routing_hints=use_routing_hints)
    if router_mode == "model":
        router_model = _build_router_model(model_name=router_model_name or model_name)
        return route_task_with_model(routing_input, model=router_model)
    return route_task(routing_input)


def route_for_task(
    task: Task,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    router_mode: RoutingStrategy = "rules",
    router_model_name: str | None = None,
    use_routing_hints: bool = True,
) -> RouterMode:
    """Choose the execution loop from task semantics, not task ids."""
    decision = _route_decision_for_task(
        task,
        model_name=model_name,
        router_mode=router_mode,
        router_model_name=router_model_name,
        use_routing_hints=use_routing_hints,
    )
    return decision.execution_route


def run_task_router(
    task: Task,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    deep_profile: str | None = DEFAULT_ROUTER_DEEP_PROFILE,
    router_mode: RoutingStrategy = "rules",
    router_model_name: str | None = None,
    use_routing_hints: bool = True,
    keep_workspace: bool = False,
    recursion_limit: int = 80,
    agent_error_retries: int = 0,
    retry_base_delay: float = 1.0,
    action_timeout: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    action_error_retries: int = DEFAULT_ACTION_ERROR_RETRIES,
    decision: RoutingDecision | None = None,
) -> TaskRun:
    """Run one task through the routed benchmark loop."""
    resolved_decision = decision or _route_decision_for_task(
        task,
        model_name=model_name,
        router_mode=router_mode,
        router_model_name=router_model_name,
        use_routing_hints=use_routing_hints,
    )
    deep_env = None
    if deep_profile is not None:
        deep_env = _deep_agent_env(model_name=model_name, deep_profile=deep_profile)
    return run_routed(
        task,
        decision=resolved_decision,
        direct_runner=run_task_direct,
        deep_runner=run_task,
        direct_kwargs={
            "model_name": model_name,
            "keep_workspace": keep_workspace,
            "action_timeout": action_timeout,
            "max_actions": max_actions,
            "action_error_retries": action_error_retries,
        },
        deep_kwargs={
            "keep_workspace": keep_workspace,
            "recursion_limit": recursion_limit,
            "agent_error_retries": agent_error_retries,
            "retry_base_delay": retry_base_delay,
            "correction_retries": 0,
            "recursion_recovery_attempts": 0,
            "finalization_retries": 0,
        },
        deep_env=deep_env,
    )


def run_all_router(
    task_ids: list[str] | None = None,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    deep_profile: str = DEFAULT_ROUTER_DEEP_PROFILE,
    router_mode: RoutingStrategy = "rules",
    router_model_name: str | None = None,
    use_routing_hints: bool = True,
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
                router_mode=router_mode,
                router_model_name=router_model_name,
                use_routing_hints=use_routing_hints,
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
            router_mode=router_mode,
            router_model_name=router_model_name,
            use_routing_hints=use_routing_hints,
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
    router_mode: RoutingStrategy,
    router_model_name: str | None,
    use_routing_hints: bool,
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
        decision = _route_decision_for_task(
            task,
            model_name=model_name,
            router_mode=router_mode,
            router_model_name=router_model_name,
            use_routing_hints=use_routing_hints,
        )
        print(f"-> {task.id}: {task.name} [{decision.mode_label}]")
        run = run_task_router(
            task,
            model_name=model_name,
            deep_profile=None,
            router_mode=router_mode,
            router_model_name=router_model_name,
            use_routing_hints=use_routing_hints,
            keep_workspace=keep_workspace,
            recursion_limit=recursion_limit,
            agent_error_retries=agent_error_retries,
            retry_base_delay=retry_base_delay,
            action_timeout=action_timeout,
            max_actions=max_actions,
            action_error_retries=action_error_retries,
            decision=decision,
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
    router_mode: RoutingStrategy,
    router_model_name: str | None,
    use_routing_hints: bool,
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
    task_decisions = {
        task.id: _route_decision_for_task(
            task,
            model_name=model_name,
            router_mode=router_mode,
            router_model_name=router_model_name,
            use_routing_hints=use_routing_hints,
        )
        for task in tasks
    }
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                run_task_router,
                task,
                model_name=model_name,
                deep_profile=None,
                router_mode=router_mode,
                router_model_name=router_model_name,
                use_routing_hints=use_routing_hints,
                keep_workspace=keep_workspace,
                recursion_limit=recursion_limit,
                agent_error_retries=agent_error_retries,
                retry_base_delay=retry_base_delay,
                action_timeout=action_timeout,
                max_actions=max_actions,
                action_error_retries=action_error_retries,
                decision=task_decisions[task.id],
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
                    f"{task_decisions[task.id].mode_label:13s} "
                    f"{run.elapsed_seconds:5.1f}s - {_one_line_detail(run)}"
                )
                if keep_workspace and run.workspace:
                    print(f"           workspace: {run.workspace}")
    results.sort(key=lambda r: _task_sort_key(r.task_id))
    return results
