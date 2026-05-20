"""Public workspace runtime built on top of the shared routing policy."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents_gigachat.orchestrator import build_deep_agent_env, run_routed, temporary_env
from deepagents_gigachat.routing import (
    RoutingDecision,
    RoutingStrategy,
    ToolRoute,
    build_routing_input,
    classify_tool_route,
    route_task,
    route_task_with_model,
)

DEFAULT_MODEL_NAME = "GigaChat-3-Ultra"
DEFAULT_RECURSION_LIMIT = 80
DEFAULT_ACTION_TIMEOUT_SECONDS = 120
DEFAULT_MAX_ACTIONS = 2
DEFAULT_ACTION_ERROR_RETRIES = 2


@dataclass(slots=True)
class RoutedInvocationResult:
    """The result of one routed workspace invocation."""

    decision: RoutingDecision
    workspace: Path
    elapsed_seconds: float
    deep_state: Any | None = None
    controller_response: str | None = None
    controller_action: dict[str, str] | None = None
    action_result: subprocess.CompletedProcess[str] | None = None


def load_env_from_dotenv(search_from: str | Path | None = None) -> None:
    """Best-effort load of a nearby `.env` file."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    candidates: list[Path] = []
    if search_from is not None:
        start = Path(search_from).resolve()
        candidates.extend(
            [
                start if start.name == ".env" else start / ".env",
                start.parent / ".env",
            ]
        )

    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            repo_root / ".env",
            repo_root.parent / ".env",
            Path.cwd() / ".env",
            Path.cwd().parent / ".env",
        ]
    )

    seen: set[Path] = set()
    for env_path in candidates:
        normalized = env_path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            load_dotenv(normalized, override=False)
            return


def ensure_credentials() -> None:
    """Raise when no GigaChat credentials are configured."""
    if os.getenv("GIGACHAT_CREDENTIALS"):
        return
    if os.getenv("GIGACHAT_USER") and os.getenv("GIGACHAT_PASSWORD"):
        return
    raise SystemExit(
        "GigaChat credentials are not configured. "
        "Set GIGACHAT_CREDENTIALS or the GIGACHAT_USER + GIGACHAT_PASSWORD pair."
    )


def build_model(*, model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Construct a `langchain-gigachat` chat model."""
    from langchain_gigachat import GigaChat

    return GigaChat(
        model=model_name,
        base_url=os.getenv("GIGACHAT_BASE_URL", "https://gigachat.sberdevices.ru/v1"),
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=600,
    )


def build_deep_agent(
    workspace: str | Path,
    *,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    model_name: str = DEFAULT_MODEL_NAME,
    deep_profile: str | None = None,
    tools: Sequence[Any] | None = None,
    backend_timeout: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
) -> Any:
    """Build a deep agent for an arbitrary local workspace."""
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend

    from deepagents_gigachat.harness_profile import register_harness
    from deepagents_gigachat.pi_tools import build_pi_like_tools

    workspace_path = Path(workspace).resolve()
    env = build_deep_agent_env(model_name=model_name, deep_profile=deep_profile)
    with temporary_env(env):
        register_harness()
        backend = LocalShellBackend(
            root_dir=workspace_path,
            virtual_mode=True,
            timeout=backend_timeout,
            inherit_env=True,
        )
        resolved_tools = list(tools) if tools is not None else None
        if resolved_tools is None and os.getenv("DEEPAGENTS_GIGACHAT_PROFILE", "").strip().lower() == "pi-tools":
            resolved_tools = build_pi_like_tools(workspace_path)
        agent = create_deep_agent(
            model=build_model(model_name=model_name),
            backend=backend,
            tools=resolved_tools,
        )
    return agent.with_config({"recursion_limit": recursion_limit})


def invoke_deep(
    prompt: str,
    *,
    workspace: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    deep_profile: str | None = None,
    tools: Sequence[Any] | None = None,
    load_env: bool = False,
    ensure_auth: bool = False,
) -> RoutedInvocationResult:
    """Run one prompt through the deepagents branch in a local workspace."""
    return _invoke_deep_impl(
        prompt,
        workspace=workspace,
        model_name=model_name,
        recursion_limit=recursion_limit,
        deep_profile=deep_profile,
        tools=tools,
        decision=None,
        load_env=load_env,
        ensure_auth=ensure_auth,
    )


def _invoke_deep_impl(
    prompt: str,
    *,
    workspace: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    deep_profile: str | None = None,
    tools: Sequence[Any] | None = None,
    decision: RoutingDecision | None = None,
    load_env: bool = False,
    ensure_auth: bool = False,
) -> RoutedInvocationResult:
    """Internal deep branch entrypoint with optional precomputed decision."""
    workspace_path = Path(workspace).resolve()
    if load_env:
        load_env_from_dotenv(workspace_path)
    if ensure_auth:
        ensure_credentials()

    started = time.monotonic()
    resolved_decision = decision or RoutingDecision(
        execution_route="deep",
        tool_route=classify_tool_route(prompt),
    )
    agent = build_deep_agent(
        workspace_path,
        recursion_limit=recursion_limit,
        model_name=model_name,
        deep_profile=deep_profile,
        tools=tools,
    )
    deep_state = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return RoutedInvocationResult(
        decision=resolved_decision,
        workspace=workspace_path,
        elapsed_seconds=time.monotonic() - started,
        deep_state=deep_state,
    )


def invoke_direct(
    prompt: str,
    *,
    workspace: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    load_env: bool = False,
    ensure_auth: bool = False,
) -> RoutedInvocationResult:
    """Run one prompt through the compact direct controller branch."""
    return _invoke_direct_impl(
        prompt,
        workspace=workspace,
        model_name=model_name,
        action_timeout=DEFAULT_ACTION_TIMEOUT_SECONDS,
        max_actions=DEFAULT_MAX_ACTIONS,
        action_error_retries=DEFAULT_ACTION_ERROR_RETRIES,
        required_artifacts=None,
        decision=None,
        load_env=load_env,
        ensure_auth=ensure_auth,
    )


def _invoke_direct_impl(
    prompt: str,
    *,
    workspace: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    action_timeout: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    action_error_retries: int = DEFAULT_ACTION_ERROR_RETRIES,
    required_artifacts: Iterable[str] | None = None,
    decision: RoutingDecision | None = None,
    load_env: bool = False,
    ensure_auth: bool = False,
) -> RoutedInvocationResult:
    """Internal direct branch entrypoint with controller tuning knobs."""
    workspace_path = Path(workspace).resolve()
    if load_env:
        load_env_from_dotenv(workspace_path)
    if ensure_auth:
        ensure_credentials()

    started = time.monotonic()
    resolved_decision = decision or RoutingDecision(
        execution_route="direct",
        tool_route=classify_tool_route(prompt),
    )
    model = build_model(model_name=model_name)
    action_budget = max(max_actions, 1)
    error_budget = max(action_error_retries, 0)
    expected_artifacts = _resolved_required_artifacts(prompt, required_artifacts)

    action_result: subprocess.CompletedProcess[str] | None = None
    action: dict[str, str] | None = None
    response_text: str | None = None
    previous_failure: str | None = None
    missing_artifacts = _missing_required_artifacts(expected_artifacts, workspace_path)
    completed_actions = 0
    error_attempts = 0

    while completed_actions < action_budget:
        try:
            response = model.invoke(
                _controller_messages(
                    prompt,
                    route=resolved_decision.tool_route,
                    missing_artifacts=missing_artifacts or None,
                    previous_failure=previous_failure,
                )
            )
            response_text = _message_content(response)
            action = _parse_action(response_text)
            _preflight_action(action)
            action_result = _run_action(
                action,
                workspace=workspace_path,
                timeout=action_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - bubble after bounded retries
            if error_attempts >= error_budget:
                raise
            error_attempts += 1
            previous_failure = _controller_failure_detail(exc)
            missing_artifacts = []
            continue

        if action_result.returncode != 0:
            if error_attempts >= error_budget:
                raise RuntimeError(_action_failure_detail(action_result))
            error_attempts += 1
            previous_failure = _action_failure_detail(action_result)
            missing_artifacts = []
            continue

        completed_actions += 1
        previous_failure = None
        missing_artifacts = _missing_required_artifacts(expected_artifacts, workspace_path)
        if not missing_artifacts:
            return RoutedInvocationResult(
                decision=resolved_decision,
                workspace=workspace_path,
                elapsed_seconds=time.monotonic() - started,
                controller_response=response_text,
                controller_action=action,
                action_result=action_result,
            )

    raise FileNotFoundError(
        "Direct controller finished without creating required artifacts: "
        f"{', '.join(missing_artifacts)}"
    )


def invoke_routed(
    prompt: str,
    *,
    workspace: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    router_mode: RoutingStrategy = "rules",
    router_model_name: str | None = None,
    deep_profile: str = "hybrid",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    tools: Sequence[Any] | None = None,
    load_env: bool = True,
    ensure_auth: bool = True,
) -> RoutedInvocationResult:
    """Route one prompt between the direct controller and deepagents."""
    workspace_path = Path(workspace).resolve()
    if load_env:
        load_env_from_dotenv(workspace_path)
    if ensure_auth:
        ensure_credentials()

    routing_input = build_routing_input(prompt)
    if router_mode == "model":
        router_model = build_model(model_name=router_model_name or model_name)
        decision = route_task_with_model(routing_input, model=router_model)
    else:
        decision = route_task(routing_input)
    return run_routed(
        prompt,
        decision=decision,
        direct_runner=_invoke_direct_impl,
        deep_runner=_invoke_deep_impl,
        direct_kwargs={
            "workspace": workspace_path,
            "model_name": model_name,
            "decision": decision,
            "load_env": False,
            "ensure_auth": False,
        },
        deep_kwargs={
            "workspace": workspace_path,
            "model_name": model_name,
            "recursion_limit": recursion_limit,
            "deep_profile": None,
            "tools": tools,
            "decision": decision,
            "load_env": False,
            "ensure_auth": False,
        },
        deep_env=build_deep_agent_env(
            model_name=model_name,
            deep_profile=deep_profile,
        ),
    )


def _controller_messages(
    prompt: str,
    *,
    route: ToolRoute,
    missing_artifacts: list[str] | None = None,
    previous_failure: str | None = None,
) -> list[dict[str, str]]:
    user_content = prompt
    if previous_failure:
        user_content = (
            f"{user_content}\n\n"
            "The previous action attempt failed:\n"
            f"{previous_failure}\n\n"
            "Return one corrected JSON action that completes the original "
            "request. Do not repeat the same failing approach."
        )
    if missing_artifacts:
        user_content = (
            f"{user_content}\n\n"
            "The previous action ran, but these required artifacts are still "
            f"missing: {', '.join(missing_artifacts)}. Return one corrected "
            "JSON action that creates them and completes the original request."
        )
    return [
        {
            "role": "system",
            "content": _controller_system_prompt(route),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _controller_system_prompt(route: ToolRoute) -> str:
    route_hint = {
        "data": (
            "This is a data/artifact task. Prefer kind='python'. Parse source "
            "files and write the requested output artifact(s) exactly."
        ),
        "search": (
            "This is a search/extraction task. Prefer kind='python'. Scan all "
            "requested files, process every match, and strip prefixes unless "
            "the request explicitly asks for file names or line numbers."
        ),
        "filesystem": (
            "This is a filesystem state task. Prefer kind='shell' for mv/rm/"
            "mkdir -p or kind='python' with pathlib. After rename/move/delete, "
            "the old path must be gone."
        ),
        "hybrid": (
            "This is a general coding task. Use one minimal Python or shell "
            "action that changes the workspace exactly as requested."
        ),
    }[route]
    return (
        "You are a direct coding controller.\n"
        "Return exactly one JSON object and no markdown, no prose.\n"
        "Schema options:\n"
        '{"kind":"python","code":"..."}\n'
        '{"kind":"shell","cmd":"..."}\n'
        "The action will run with cwd set to the workspace. Use relative paths "
        "only. Do not use host paths like /Users/...\n"
        "The action must create, update, move, or delete the requested files "
        "exactly, then exit.\n"
        "If the request mentions multiple files or output artifacts, create "
        "all of them in the same action.\n"
        "Prefer Python stdlib and already-installed packages. For structured "
        "data tasks, write the final artifact from Python instead of copying "
        "shell output manually.\n"
        "For grep/search outputs, write only the requested values. Strip path "
        "and line-number prefixes unless explicitly requested.\n"
        "For counts and integer totals, write integer strings like 290, not 290.0.\n"
        "For CSV-like outputs, preserve header order, row order unless the "
        "request asks for sorting, and exact delimiters.\n"
        "Use only one bounded action. Do not ask questions.\n"
        f"{route_hint}"
    )


def _message_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    return str(content)


def _parse_action(text: str) -> dict[str, str]:
    payload = _extract_json(text)
    action = json.loads(payload)
    if not isinstance(action, dict):
        raise ValueError("controller response must be a JSON object")
    kind = action.get("kind")
    if kind == "python" and isinstance(action.get("code"), str):
        return {"kind": "python", "code": action["code"]}
    if kind == "shell" and isinstance(action.get("cmd"), str):
        return {"kind": "shell", "cmd": action["cmd"]}
    raise ValueError("action must be {'kind':'python','code':...} or {'kind':'shell','cmd':...}")


def _preflight_action(action: dict[str, str]) -> None:
    if action["kind"] != "python":
        return
    try:
        ast.parse(action["code"])
    except SyntaxError as exc:
        raise ValueError(_python_syntax_error_detail(exc)) from exc


def _python_syntax_error_detail(exc: SyntaxError) -> str:
    location = f"line {exc.lineno}"
    if exc.offset is not None:
        location += f", column {exc.offset}"
    text = (exc.text or "").strip()
    suffix = f": {text!r}" if text else ""
    return f"Python action has SyntaxError at {location}: {exc.msg}{suffix}"


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        candidate = stripped[match.start() :]
        for payload in _json_payload_candidates(candidate):
            try:
                _, end = decoder.raw_decode(payload)
            except json.JSONDecodeError:
                continue
            return payload[:end]
    python_literal_payload = _extract_python_literal_payload(stripped)
    if python_literal_payload is not None:
        return python_literal_payload
    raise ValueError(f"no JSON object found in controller response: {text!r}")


def _json_payload_candidates(text: str) -> tuple[str, ...]:
    repaired = text.replace("\\'", "'")
    if repaired == text:
        return (text,)
    return (text, repaired)


def _extract_python_literal_payload(text: str) -> str | None:
    for match in re.finditer(r"\{", text):
        prefix = _balanced_brace_prefix(text[match.start() :])
        if prefix is None:
            continue
        for payload in _json_payload_candidates(prefix):
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, dict):
                return json.dumps(parsed)
    return None


def _balanced_brace_prefix(text: str) -> str | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return None


def _run_action(
    action: dict[str, str],
    *,
    workspace: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    if action["kind"] == "python":
        argv = [sys.executable, "-c", action["code"]]
    else:
        argv = ["/bin/bash", "-lc", action["cmd"]]
    return subprocess.run(  # noqa: S603 - trusted local workspace command
        argv,
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
    )


def _controller_failure_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _action_failure_detail(action_result: subprocess.CompletedProcess[str]) -> str:
    tail = f"{action_result.stderr}\n{action_result.stdout}".strip()[-600:]
    return f"Action exited with {action_result.returncode}. Output tail: {tail!r}"


def _resolved_required_artifacts(
    prompt: str,
    required_artifacts: Iterable[str] | None,
) -> list[str]:
    if required_artifacts is not None:
        return [artifact.lstrip("./") for artifact in required_artifacts]
    return _named_output_artifacts(prompt)


def _missing_required_artifacts(required_artifacts: list[str], workspace: Path) -> list[str]:
    return [artifact for artifact in required_artifacts if not (workspace / artifact).exists()]


def _named_output_artifacts(prompt: str) -> list[str]:
    patterns = (
        r"(?:создай|создать)\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:сделай|сделать)\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:в|во)\s+один\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:запиши|записать|сохрани|сохранить)\b[^.\n]*?\bв\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:write|create|produce|generate)\s+(?:file\s+)?(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:to|into)\s+one\s+file\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:save|write)\b[^.\n]*?\b(?:to|into)\s+(?:file\s+)?(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
    )
    seen: set[str] = set()
    artifacts: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            artifact = match.group("path").lstrip("./")
            if artifact not in seen:
                seen.add(artifact)
                artifacts.append(artifact)
    return artifacts


__all__ = [
    "DEFAULT_ACTION_ERROR_RETRIES",
    "DEFAULT_ACTION_TIMEOUT_SECONDS",
    "DEFAULT_MAX_ACTIONS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_RECURSION_LIMIT",
    "RoutedInvocationResult",
    "build_deep_agent",
    "build_model",
    "ensure_credentials",
    "invoke_deep",
    "invoke_direct",
    "invoke_routed",
    "load_env_from_dotenv",
]
