"""Run benchmark tasks through a small direct one-action GigaChat controller."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from deepagents_gigachat.routing import classify_tool_route
from harness_bench.core import Task
from harness_bench.runner import (
    TaskRun,
    _load_env_from_dotenv,
    _one_line_detail,
    _task_sort_key,
)
from harness_bench.tasks import ALL_TASKS, get_task

DEFAULT_DIRECT_MODEL = "GigaChat-3-Ultra"
DEFAULT_ACTION_TIMEOUT_SECONDS = 120
DEFAULT_MAX_ACTIONS = 2
DEFAULT_ACTION_ERROR_RETRIES = 2


def _ensure_credentials() -> None:
    if os.getenv("GIGACHAT_CREDENTIALS"):
        return
    if os.getenv("GIGACHAT_USER") and os.getenv("GIGACHAT_PASSWORD"):
        return
    raise SystemExit(
        "Не заданы учётные данные GigaChat. "
        "Укажи GIGACHAT_CREDENTIALS либо пару GIGACHAT_USER + GIGACHAT_PASSWORD."
    )


def _build_model(*, model_name: str = DEFAULT_DIRECT_MODEL) -> Any:
    from langchain_gigachat import GigaChat

    return GigaChat(
        model=model_name,
        base_url=os.getenv("GIGACHAT_BASE_URL", "https://gigachat.sberdevices.ru/v1"),
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=600,
    )


def run_task_direct(
    task: Task,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    keep_workspace: bool = False,
    action_timeout: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    action_error_retries: int = DEFAULT_ACTION_ERROR_RETRIES,
) -> TaskRun:
    """Run one task with a bounded model-chosen shell/Python controller."""
    workspace_keepalive: TemporaryDirectory | None = None
    try:
        if keep_workspace:
            workspace_path = Path(
                __import__("tempfile").mkdtemp(prefix=f"hb_direct_{task.id}_")
            )
        else:
            workspace_keepalive = TemporaryDirectory(prefix=f"hb_direct_{task.id}_")
            workspace_path = Path(workspace_keepalive.name)

        task.setup(workspace_path)
        started = time.monotonic()
        try:
            model = _build_model(model_name=model_name)
            action_result: subprocess.CompletedProcess[str] | None = None
            missing_artifacts: list[str] = []
            previous_failure: str | None = None
            completed_actions = 0
            error_attempts = 0
            action_budget = max(max_actions, 1)
            error_budget = max(action_error_retries, 0)
            while completed_actions < action_budget:
                try:
                    response = model.invoke(
                        _controller_messages(
                            task.prompt,
                            missing_artifacts=missing_artifacts or None,
                            previous_failure=previous_failure,
                        )
                    )
                    action = _parse_action(_message_content(response))
                    _preflight_action(action)
                    action_result = _run_action(
                        action,
                        workspace=workspace_path,
                        timeout=action_timeout,
                    )
                except Exception as exc:  # noqa: BLE001 - retry invalid controller turns
                    if error_attempts >= error_budget:
                        raise
                    error_attempts += 1
                    previous_failure = _controller_failure_detail(exc)
                    missing_artifacts = []
                    continue

                if action_result.returncode != 0:
                    if error_attempts >= error_budget:
                        break
                    error_attempts += 1
                    previous_failure = _action_failure_detail(action_result)
                    missing_artifacts = []
                    continue

                completed_actions += 1
                missing_artifacts = _missing_named_artifacts(task.prompt, workspace_path)
                format_issue = _format_issue_detail(task.prompt, workspace_path)
                if format_issue and completed_actions < action_budget:
                    previous_failure = format_issue
                    missing_artifacts = []
                    continue
                if not missing_artifacts:
                    break
                previous_failure = None
            if action_result is None:
                raise RuntimeError("direct controller did not run an action")
        except Exception:  # noqa: BLE001 - surface as benchmark failure
            return TaskRun(
                task_id=task.id,
                passed=False,
                message="",
                elapsed_seconds=time.monotonic() - started,
                error=traceback.format_exc(),
                workspace=workspace_path if keep_workspace else None,
            )

        outcome = task.verify(workspace_path)
        message = outcome.message
        if not outcome.passed and action_result.returncode != 0:
            tail = f"{action_result.stderr}\n{action_result.stdout}".strip()[-300:]
            message = f"{outcome.message} | action exit={action_result.returncode}: {tail!r}"
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


def _controller_messages(
    prompt: str,
    *,
    missing_artifacts: list[str] | None = None,
    previous_failure: str | None = None,
) -> list[dict[str, str]]:
    route = classify_tool_route(prompt)
    user_content = prompt
    if previous_failure:
        user_content = (
            f"{user_content}\n\n"
            "The previous action attempt failed before verification:\n"
            f"{previous_failure}\n\n"
            "Return one corrected JSON action that completes the original "
            "request. Do not repeat the failed approach if the failure shows "
            "a missing package, invalid JSON shape, or command/runtime error."
        )
    if missing_artifacts:
        user_content = (
            f"{user_content}\n\n"
            "The previous action ran, but these named output artifacts are "
            f"still missing: {', '.join(missing_artifacts)}. Return one JSON "
            "action that creates the missing artifact(s) and completes the "
            "original request. Do not change already-correct files unless "
            "needed."
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


def _controller_system_prompt(route: str) -> str:
    route_hint = {
        "data": (
            "This is a data/artifact task. Prefer kind='python'. Parse input "
            "files with Python libraries and write the requested output file(s)."
        ),
        "search": (
            "This is a search/extraction task. Prefer kind='python'. Scan all "
            "requested files, process every match, strip prefixes unless paths "
            "are requested, and ignore wc summary labels like total."
        ),
        "filesystem": (
            "This is a filesystem commit task. Prefer kind='shell' for mv/rm/"
            "mkdir -p or kind='python' with pathlib. Preserve original file "
            "contents when moving, renaming, or converting files into package "
            "directories, and ensure old paths are gone."
        ),
        "hybrid": (
            "This is a general coding task. Use a small Python or shell action "
            "to edit files or run a narrow test if useful."
        ),
    }.get(route, "")
    return (
        "You are a direct benchmark controller.\n"
        "Return exactly one JSON object and no markdown, no prose.\n"
        "Schema options:\n"
        '{"kind":"python","code":"..."}\n'
        '{"kind":"shell","cmd":"..."}\n'
        "The action will run with cwd set to the task workspace. Use relative "
        "paths only. Do not use host paths. The action must create/update/move/"
        "delete the requested files exactly, then exit.\n"
        "If the request names multiple files or asks to write a script and run "
        "it, create every named artifact, not only the final output.\n"
        "If the prompt gives exact literal lines or says to write exactly N "
        "non-empty lines, write only those lines in that order, preserving "
        "indentation. Do not add useful-looking schema completions, examples, "
        "comments, blank sections, or extra config entries.\n"
        "For filesystem moves/renames/conversions, read or move the original "
        "content into the requested destination before deleting the source. "
        "Do not replace a non-empty source file with an empty destination file "
        "such as an empty __init__.py.\n"
        "Prefer Python stdlib and already-installed packages. Do not import "
        "optional writer packages unless you know they are installed; for "
        "small TOML/INI/YAML/config edits, manual text-preserving edits are "
        "often safer.\n"
        "When reading CSV/table files with headers, use a header-aware parser "
        "or explicitly skip the header row before numeric conversion.\n"
        "When editing CSV, preserve row order, column order, and integer "
        "formatting. Prefer csv.DictReader/DictWriter over pandas when pandas "
        "would turn integer arithmetic results into values like 300.0.\n"
        "For numeric tables, preserve the apparent input scalar types in CSV "
        "outputs: write integers as integers, and write floats only when the "
        "computed value is genuinely fractional or the prompt asks for floats.\n"
        "For median, prefer Python statistics.median; over an even number of "
        "values use the average of the two middle sorted values, not either "
        "middle value alone. Do not compute even-length median as "
        "sorted_values[len(sorted_values)//2].\n"
        "For pivot/groupby tables, use deterministic row and column ordering "
        "from the input or natural sorted order when the prompt does not "
        "specify an order; fill absent numeric cells with 0 rather than blanks "
        "unless the prompt says otherwise.\n"
        "When converting XLSX to CSV/JSON, preserve sheet/header order and "
        "cell scalar types; write CSV with comma delimiters, exactly one "
        "header row, no index column, no blank rows, and no integer values "
        "stringified as 1.0.\n"
        "For Apache/common access logs, do not use naive line.split() to "
        "locate the HTTP status because the quoted request contains spaces. "
        "Parse the quoted request as one field or use a regex; the HTTP "
        "status is the integer immediately after the closing quote, not the "
        "URL path inside the request.\n"
        "For Apache/common log grouping by hour, extract HH from timestamps "
        "like [DD/Mon/YYYY:HH:MM:SS +ZZZZ]; group by those two hour digits "
        "from every line, not by date and not by the request URL.\n"
        "For Apache/common log filtering by HTTP status, match the status "
        "integer immediately after the closing quote and write the entire "
        "original matching line byte-for-byte in input order.\n"
        "For search tasks, scan every regular file under the requested path "
        "unless the prompt explicitly restricts extensions; do not stop after "
        "the first matching file.\n"
        "When counting occurrences of a specific character or letter, count "
        "exact single-character occurrences in the whole file, not words or "
        "matching lines; respect lowercase/uppercase unless the prompt says "
        "otherwise.\n"
        "For word-count tasks, count whitespace-separated tokens in the whole "
        "input unless the prompt defines a different unit.\n"
        "For counts and whole-number totals, write an integer string without "
        "a decimal suffix, for example 290 not 290.0.\n"
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
        raise ValueError("direct controller response must be a JSON object")
    kind = action.get("kind")
    if kind == "python" and isinstance(action.get("code"), str):
        return {"kind": "python", "code": action["code"]}
    if kind == "shell" and isinstance(action.get("cmd"), str):
        return {"kind": "shell", "cmd": action["cmd"]}
    raise ValueError("action must be {'kind':'python','code':...} or {'kind':'shell','cmd':...}")


def _preflight_action(action: dict[str, str]) -> None:
    """Catch cheap syntax errors before spending an execution attempt."""
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
    hint = ""
    if text.startswith("with ") and text.endswith(","):
        hint = (
            " For multiple context managers, use one valid `with a, b:` line, "
            "wrap them in `with (...):`, or use nested with statements; do not "
            "leave a bare trailing comma at the end of the with line."
        )
    if ("r'" in text or 'r"' in text) and "re." in text:
        hint += (
            " If a regex pattern contains quote characters, use the opposite "
            "quote style or a triple-quoted raw string for the pattern."
        )
    return f"Python action has SyntaxError at {location}: {exc.msg}{suffix}{hint}"


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
    return subprocess.run(  # noqa: S603 - trusted local benchmark experiment
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


def _missing_named_artifacts(prompt: str, workspace: Path) -> list[str]:
    """Return requested output artifacts that are not present after an action."""
    artifacts = _named_output_artifacts(prompt)
    return [artifact for artifact in artifacts if not (workspace / artifact).exists()]


def _named_output_artifacts(prompt: str) -> list[str]:
    """Best-effort extraction of output filenames from English/Russian prompts."""
    patterns = (
        r"(?:создай|создать)\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:сделай|сделать)\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:в|во)\s+один\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:запиши|записать|сохрани|сохранить)\b[^.\n]*?\bв\s+файл\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
        r"(?:запиши|записать|сохрани|сохранить)\b[^.\n]*?\bв\s+(?:том\s+же\s+)?файле\s+(?P<path>[\w./-]+\.[A-Za-z0-9]+)",
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


def _format_issue_detail(prompt: str, workspace: Path) -> str | None:
    """Detect simple prompt-compliance issues without using verifier feedback."""
    status_filter_issue = _log_status_filter_issue_detail(prompt, workspace)
    if status_filter_issue:
        return status_filter_issue

    if not _expects_integerish_output(prompt):
        return None

    offenders = []
    for artifact in _named_output_artifacts(prompt):
        path = workspace / artifact
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if re.search(r"(?<![\w.])-?\d+\.0(?![\w.])", text):
            offenders.append(artifact)

    if not offenders:
        return None
    return (
        "Output formatting issue: "
        f"{', '.join(offenders)} contains integer-like decimal suffixes such "
        "as 300.0 or 10.0. Return one corrected action that rewrites copied "
        "integer input cells and integer-valued result cells as integer "
        "strings, for example 10 not 10.0, without changing requested columns "
        "or row order. Keep genuinely fractional metrics as decimals, and keep "
        "prompt-requested decimal formatting where the prompt explicitly asks "
        "for it. If using Python, convert integer-valued floats with int(x) "
        "before writing integer-style cells."
    )


def _expects_integerish_output(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = (
        "count",
        "integer",
        "total",
        "whole-number",
        "количество",
        "посчитай",
        "произвед",
        "столбец total",
        "целое",
        "число",
        "значени",
        "values",
    )
    return any(marker in lowered for marker in markers)


def _log_status_filter_issue_detail(prompt: str, workspace: Path) -> str | None:
    status = _requested_http_status_filter(prompt)
    if status is None:
        return None

    log_path = workspace / "access.log"
    if not log_path.is_file():
        return None

    output_logs = [
        workspace / artifact
        for artifact in _named_output_artifacts(prompt)
        if artifact.endswith(".log")
    ]
    if not output_logs:
        return None

    try:
        input_lines = log_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    expected = [
        line
        for line in input_lines
        if (match := re.search(r'"[^"]*"\s+(\d{3})\b', line))
        and match.group(1) == status
    ]
    if not expected:
        return None

    offenders = []
    for output_log in output_logs:
        if not output_log.is_file():
            continue
        try:
            actual = output_log.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        if actual != expected:
            offenders.append(output_log.name)

    if not offenders:
        return None
    return (
        "Apache log status filter issue: "
        f"{', '.join(offenders)} does not contain exactly the original input "
        f"lines whose HTTP status after the quoted request is {status}. Return "
        "one corrected action that parses the status immediately after the "
        "closing quote, preserves whole matching lines byte-for-byte, and "
        "keeps their input order."
    )


def _requested_http_status_filter(prompt: str) -> str | None:
    lowered = prompt.lower()
    if "http" not in lowered or "status" not in lowered and "статус" not in lowered:
        return None
    patterns = (
        r"(?:статус|status)[^.\n]{0,120}(?:равен|equals|=)\s*(?P<status>\d{3})",
        r"(?P<status>\d{3})[^.\n]{0,120}(?:status|статус)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group("status")
    return None


def run_all_direct(
    task_ids: list[str] | None = None,
    *,
    model_name: str = DEFAULT_DIRECT_MODEL,
    keep_workspace: bool = False,
    action_timeout: int = DEFAULT_ACTION_TIMEOUT_SECONDS,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    action_error_retries: int = DEFAULT_ACTION_ERROR_RETRIES,
    concurrency: int = 1,
) -> list[TaskRun]:
    _load_env_from_dotenv()
    _ensure_credentials()

    targets = [get_task(tid) for tid in task_ids] if task_ids else list(ALL_TASKS)

    if concurrency <= 1:
        results: list[TaskRun] = []
        for task in targets:
            print(f"-> {task.id}: {task.name}")
            run = run_task_direct(
                task,
                model_name=model_name,
                keep_workspace=keep_workspace,
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

    print_lock = threading.Lock()
    completed = 0
    total = len(targets)
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                run_task_direct,
                task,
                model_name=model_name,
                keep_workspace=keep_workspace,
                action_timeout=action_timeout,
                max_actions=max_actions,
                action_error_retries=action_error_retries,
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
                    f"{run.elapsed_seconds:5.1f}s - {_one_line_detail(run)}"
                )
                if keep_workspace and run.workspace:
                    print(f"           workspace: {run.workspace}")
    results.sort(key=lambda r: _task_sort_key(r.task_id))
    return results
