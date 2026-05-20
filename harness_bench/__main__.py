"""Command-line entry point for the benchmark.

Examples:

    # List all tasks
    uv run python -m harness_bench list

    # Run the whole benchmark against GigaChat (uses .env for credentials)
    uv run python -m harness_bench run

    # Run a couple of tasks and keep the temp workspaces for inspection
    uv run python -m harness_bench run --task task_01_create_hello --task task_06_toggle_debug --keep

    # Run the benchmark through pi-mono (pi CLI)
    uv run python -m harness_bench run-pi --model sonnet --thinking high

    # Run the benchmark against OpenAI directly
    uv run python -m harness_bench run-openai --model gpt-4.1-mini

    # Sanity-check verifiers without any LLM calls
    uv run python -m harness_bench verify-gold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness_bench.runner import run_all, summarize, verify_gold
from harness_bench.runner_cli import (
    DEFAULT_CLI_COMMAND,
    DEFAULT_PI_ENV,
    DEFAULT_TIMEOUT_SECONDS,
    build_pi_cli_command,
    run_all_cli,
)
from harness_bench.runner_direct import DEFAULT_DIRECT_MODEL, run_all_direct
from harness_bench.runner_openai import DEFAULT_OPENAI_MODEL
from harness_bench.runner_openai import run_all as run_all_openai
from harness_bench.runner_openrouter import DEFAULT_OPENROUTER_MODEL
from harness_bench.runner_openrouter import run_all as run_all_openrouter
from harness_bench.runner_router import DEFAULT_ROUTER_DEEP_PROFILE, run_all_router
from harness_bench.tasks import ALL_TASKS


def _cmd_list(_args: argparse.Namespace) -> int:
    for task in ALL_TASKS:
        tags = f"  [{', '.join(task.tags)}]" if task.tags else ""
        print(f"  {task.id} — {task.name}{tags}")
    print(f"\nTotal: {len(ALL_TASKS)} tasks")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    results = run_all(
        task_ids=args.task,
        keep_workspace=args.keep,
        recursion_limit=args.recursion_limit,
        concurrency=args.concurrency,
        agent_error_retries=args.retry_agent_errors,
        retry_base_delay=args.retry_base_delay,
        correction_retries=args.correction_retries,
        recursion_recovery_attempts=args.recover_recursion,
        finalization_retries=args.finalization_retries,
    )
    summarize(results)
    return 0 if all(r.passed for r in results) else 1


def _cmd_run_openrouter(args: argparse.Namespace) -> int:
    results = run_all_openrouter(
        task_ids=args.task,
        model_name=args.model,
        keep_workspace=args.keep,
        recursion_limit=args.recursion_limit,
        concurrency=args.concurrency,
    )
    summarize(results)
    return 0 if all(r.passed for r in results) else 1


def _cmd_run_openai(args: argparse.Namespace) -> int:
    results = run_all_openai(
        task_ids=args.task,
        model_name=args.model,
        keep_workspace=args.keep,
        recursion_limit=args.recursion_limit,
        concurrency=args.concurrency,
    )
    summarize(results)
    return 0 if all(r.passed for r in results) else 1


def _cmd_run_cli(args: argparse.Namespace) -> int:
    results = run_all_cli(
        task_ids=args.task,
        cli_command=args.cli_command,
        timeout=args.timeout,
        keep_workspace=args.keep,
        concurrency=args.concurrency,
        transient_403_retries=args.retry_transient_403,
        retry_base_delay=args.retry_base_delay,
    )
    summarize(results)
    return 0 if all(r.passed for r in results) else 1


def _cmd_run_pi(args: argparse.Namespace) -> int:
    env_overrides = dict(DEFAULT_PI_ENV)
    if args.node_extra_ca_certs:
        env_overrides["NODE_EXTRA_CA_CERTS"] = str(
            Path(args.node_extra_ca_certs).expanduser().resolve()
        )
    for env_override in args.env or []:
        name, sep, value = env_override.partition("=")
        if not sep or not name:
            raise SystemExit(f"Invalid --env override: {env_override!r}")
        env_overrides[name] = value

    results = run_all_cli(
        task_ids=args.task,
        cli_command=build_pi_cli_command(
            model_name=args.model,
            provider=args.provider,
            thinking=args.thinking,
            extensions=[
                str(Path(extension).expanduser().resolve())
                for extension in args.extension or []
            ],
        ),
        timeout=args.timeout,
        keep_workspace=args.keep,
        concurrency=args.concurrency,
        env_overrides=env_overrides,
        transient_403_retries=args.retry_transient_403,
        retry_base_delay=args.retry_base_delay,
    )
    summarize(results)
    return 0 if all(r.passed for r in results) else 1


def _cmd_run_direct(args: argparse.Namespace) -> int:
    results = run_all_direct(
        task_ids=args.task,
        model_name=args.model,
        keep_workspace=args.keep,
        action_timeout=args.action_timeout,
        max_actions=args.max_actions,
        action_error_retries=args.action_error_retries,
        concurrency=args.concurrency,
    )
    summarize(results)
    return 0 if all(r.passed for r in results) else 1


def _cmd_run_router(args: argparse.Namespace) -> int:
    results = run_all_router(
        task_ids=args.task,
        model_name=args.model,
        deep_profile=args.deep_profile,
        router_mode=args.router_mode,
        router_model_name=args.router_model,
        use_routing_hints=not args.no_routing_hints,
        keep_workspace=args.keep,
        recursion_limit=args.recursion_limit,
        agent_error_retries=args.retry_agent_errors,
        retry_base_delay=args.retry_base_delay,
        action_timeout=args.action_timeout,
        max_actions=args.max_actions,
        action_error_retries=args.action_error_retries,
        concurrency=args.concurrency,
    )
    summarize(results)
    return 0 if all(r.passed for r in results) else 1


def _cmd_verify_gold(args: argparse.Namespace) -> int:
    results = verify_gold(task_ids=args.task)
    failed = [r for r in results if not r.passed]
    print()
    print("=" * 64)
    print(f"Gold-verification: {len(results) - len(failed)}/{len(results)} OK")
    if failed:
        print()
        print("Verifier failures (likely bugs in the verifier or gold solution):")
        for r in failed:
            head = (r.message or "").splitlines()
            print(f"  - {r.task_id}: {head[0] if head else ''}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness_bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all benchmark tasks")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="Run benchmark with the GigaChat agent")
    p_run.add_argument(
        "--task",
        action="append",
        help="Task id (repeatable). Run all tasks if omitted.",
    )
    p_run.add_argument(
        "--keep",
        action="store_true",
        help="Keep temp workspaces for inspection",
    )
    p_run.add_argument(
        "--recursion-limit",
        type=int,
        default=80,
        help="Cap on agent loop iterations per task (default: 80)",
    )
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run up to N tasks in parallel (default: 1; uses a thread pool, "
        "each task still has its own isolated TemporaryDirectory).",
    )
    p_run.add_argument(
        "--retry-agent-errors",
        type=int,
        default=0,
        help=(
            "Retry transient model/API failures from a fresh workspace this "
            "many times (default: 0). Verifier failures and recursion-limit "
            "errors are not retried."
        ),
    )
    p_run.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        help="Initial backoff delay for --retry-agent-errors (default: 1.0).",
    )
    p_run.add_argument(
        "--correction-retries",
        type=int,
        default=0,
        help=(
            "After a verifier failure, ask the same agent to fix the existing "
            "workspace this many times before marking the task failed "
            "(default: 0)."
        ),
    )
    p_run.add_argument(
        "--recover-recursion",
        type=int,
        default=0,
        help=(
            "After GRAPH_RECURSION_LIMIT, retry in the same workspace this "
            "many times with a short recovery prompt (default: 0)."
        ),
    )
    p_run.add_argument(
        "--finalization-retries",
        type=int,
        default=0,
        help=(
            "After verifier correction still fails, run this many short fresh "
            "same-workspace finalization passes focused on writing exact "
            "requested output files (default: 0)."
        ),
    )
    p_run.set_defaults(func=_cmd_run)

    p_or = sub.add_parser(
        "run-openrouter",
        help=(
            "Run benchmark with a deepagents agent backed by an OpenRouter model "
            "(no GigaChat-specific harness profile applied)."
        ),
    )
    p_or.add_argument("--task", action="append", help="Task id (repeatable)")
    p_or.add_argument(
        "--model",
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter model id (default: {DEFAULT_OPENROUTER_MODEL}).",
    )
    p_or.add_argument("--keep", action="store_true", help="Keep temp workspaces")
    p_or.add_argument("--recursion-limit", type=int, default=80)
    p_or.add_argument("--concurrency", type=int, default=1)
    p_or.set_defaults(func=_cmd_run_openrouter)

    p_openai = sub.add_parser(
        "run-openai",
        help=(
            "Run benchmark with a deepagents agent backed by an OpenAI model "
            "(uses OPENAI_API_KEY; no GigaChat-specific harness profile applied)."
        ),
    )
    p_openai.add_argument("--task", action="append", help="Task id (repeatable)")
    p_openai.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model id (default: {DEFAULT_OPENAI_MODEL}).",
    )
    p_openai.add_argument("--keep", action="store_true", help="Keep temp workspaces")
    p_openai.add_argument("--recursion-limit", type=int, default=80)
    p_openai.add_argument("--concurrency", type=int, default=1)
    p_openai.set_defaults(func=_cmd_run_openai)

    p_gold = sub.add_parser(
        "verify-gold",
        help="Sanity-check verifiers against the gold solutions (no LLM calls)",
    )
    p_gold.add_argument("--task", action="append", help="Task id (repeatable)")
    p_gold.set_defaults(func=_cmd_verify_gold)

    p_cli = sub.add_parser(
        "run-cli",
        help=(
            "Run benchmark via an external CLI agent (default: "
            f"`{DEFAULT_CLI_COMMAND}`)."
        ),
    )
    p_cli.add_argument(
        "--task",
        action="append",
        help="Task id (repeatable). Run all tasks if omitted.",
    )
    p_cli.add_argument(
        "--keep",
        action="store_true",
        help="Keep temp workspaces for inspection",
    )
    p_cli.add_argument(
        "--cli-command",
        default=DEFAULT_CLI_COMMAND,
        help=(
            "Shell command-line prefix invoked per task. The task prompt is "
            "appended as the final argument. Default: "
            f"'{DEFAULT_CLI_COMMAND}'."
        ),
    )
    p_cli.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-task timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    p_cli.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run up to N tasks in parallel (default: 1).",
    )
    _add_cli_retry_args(p_cli)
    p_cli.set_defaults(func=_cmd_run_cli)

    p_pi = sub.add_parser(
        "run-pi",
        help=(
            "Run benchmark via the pi-mono CLI (`pi`) with clean benchmark "
            "defaults: no session reuse, no discovered context files, and no "
            "extra extensions/skills."
        ),
    )
    p_pi.add_argument(
        "--task",
        action="append",
        help="Task id (repeatable). Run all tasks if omitted.",
    )
    p_pi.add_argument(
        "--keep",
        action="store_true",
        help="Keep temp workspaces for inspection",
    )
    p_pi.add_argument(
        "--model",
        help=(
            "Pi model pattern or full provider/model id. If omitted, pi uses "
            "its configured default model."
        ),
    )
    p_pi.add_argument(
        "--provider",
        help="Explicit pi provider name (optional).",
    )
    p_pi.add_argument(
        "--thinking",
        help="Pi thinking level override (off|minimal|low|medium|high|xhigh).",
    )
    p_pi.add_argument(
        "--extension",
        action="append",
        help=(
            "Explicit pi extension path/package (repeatable). Relative paths "
            "are resolved before each task switches into its temp workspace."
        ),
    )
    p_pi.add_argument(
        "--node-extra-ca-certs",
        help=(
            "Set NODE_EXTRA_CA_CERTS for the spawned pi process, useful for "
            "GigaChat TLS roots in Node-based providers."
        ),
    )
    p_pi.add_argument(
        "--env",
        action="append",
        metavar="NAME=VALUE",
        help="Set an additional environment variable for the spawned pi process.",
    )
    p_pi.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-task timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    p_pi.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run up to N tasks in parallel (default: 1).",
    )
    _add_cli_retry_args(p_pi)
    p_pi.set_defaults(func=_cmd_run_pi)

    p_direct = sub.add_parser(
        "run-direct",
        help=(
            "Run benchmark with a direct one-action GigaChat controller "
            "(experimental, no DeepAgents graph)."
        ),
    )
    p_direct.add_argument(
        "--task",
        action="append",
        help="Task id (repeatable). Run all tasks if omitted.",
    )
    p_direct.add_argument(
        "--model",
        default=DEFAULT_DIRECT_MODEL,
        help=f"GigaChat model name (default: {DEFAULT_DIRECT_MODEL}).",
    )
    p_direct.add_argument("--keep", action="store_true", help="Keep temp workspaces")
    p_direct.add_argument(
        "--action-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for the single shell/Python action (default: 120).",
    )
    p_direct.add_argument(
        "--max-actions",
        type=int,
        default=2,
        help=(
            "Maximum bounded actions per task; extra actions are only requested "
            "when named output artifacts are still missing (default: 2)."
        ),
    )
    p_direct.add_argument(
        "--action-error-retries",
        type=int,
        default=2,
        help=(
            "Retry invalid JSON, Python syntax errors, or failed shell/Python "
            "actions before consuming another successful-action slot "
            "(default: 2)."
        ),
    )
    p_direct.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run up to N tasks in parallel (default: 1).",
    )
    p_direct.set_defaults(func=_cmd_run_direct)

    p_router = sub.add_parser(
        "run-router",
        help=(
            "Run benchmark through a semantic router: code tasks use "
            "DeepAgents/hybrid, data/search/filesystem tasks use direct."
        ),
    )
    p_router.add_argument(
        "--task",
        action="append",
        help="Task id (repeatable). Run all tasks if omitted.",
    )
    p_router.add_argument(
        "--model",
        default=DEFAULT_DIRECT_MODEL,
        help=f"GigaChat model name for both routes (default: {DEFAULT_DIRECT_MODEL}).",
    )
    p_router.add_argument(
        "--deep-profile",
        default=DEFAULT_ROUTER_DEEP_PROFILE,
        help=(
            "DEEPAGENTS_GIGACHAT_PROFILE value for deep-routed tasks "
            f"(default: {DEFAULT_ROUTER_DEEP_PROFILE})."
        ),
    )
    p_router.add_argument(
        "--router-mode",
        choices=("rules", "model"),
        default="rules",
        help="Choose the outer router implementation (default: rules).",
    )
    p_router.add_argument(
        "--router-model",
        help=(
            "Optional model name for model-based routing. Defaults to --model "
            "when --router-mode model is used."
        ),
    )
    p_router.add_argument(
        "--no-routing-hints",
        action="store_true",
        help=(
            "Ignore benchmark task tags during routing and classify tasks "
            "from prompt text alone."
        ),
    )
    p_router.add_argument("--keep", action="store_true", help="Keep temp workspaces")
    p_router.add_argument(
        "--recursion-limit",
        type=int,
        default=80,
        help="Cap on DeepAgents loop iterations per deep-routed task (default: 80).",
    )
    p_router.add_argument(
        "--retry-agent-errors",
        type=int,
        default=0,
        help=(
            "Retry transient DeepAgents model/API failures from a fresh "
            "workspace this many times (default: 0)."
        ),
    )
    p_router.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        help="Initial backoff delay for --retry-agent-errors (default: 1.0).",
    )
    p_router.add_argument(
        "--action-timeout",
        type=int,
        default=120,
        help="Timeout for direct shell/Python actions in seconds (default: 120).",
    )
    p_router.add_argument(
        "--max-actions",
        type=int,
        default=2,
        help=(
            "Maximum bounded direct actions per task; extra actions are only "
            "requested when named output artifacts are still missing "
            "(default: 2)."
        ),
    )
    p_router.add_argument(
        "--action-error-retries",
        type=int,
        default=2,
        help=(
            "Retry direct-route invalid JSON, Python syntax errors, or failed "
            "actions before consuming another successful-action slot "
            "(default: 2)."
        ),
    )
    p_router.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run up to N tasks in parallel (default: 1).",
    )
    p_router.set_defaults(func=_cmd_run_router)

    return parser


def _add_cli_retry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retry-transient-403",
        type=int,
        default=0,
        help=(
            "Retry a task this many times when the CLI exits with the known "
            "transient GigaChat streaming 403 error (default: 0)."
        ),
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=2.0,
        help=(
            "Base delay in seconds for transient 403 retries; delays grow as "
            "base, base*4, ... plus small jitter (default: 2.0)."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
