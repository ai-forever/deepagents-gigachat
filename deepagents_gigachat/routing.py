"""Shared task routing policy for GigaChat agents and controllers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

ExecutionRoute = Literal["direct", "deep"]
ToolRoute = Literal["data", "search", "filesystem", "hybrid"]

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
    "wrap top-level code into main()",
    "перенеси функцию",
    "оберни их в функцию main()",
    "if __name__ == '__main__'",
)
_CODE_FILE_MARKERS = (
    ".c",
    ".cc",
    ".cpp",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
)
_CODE_TASK_MARKERS = (
    " add ",
    " class",
    " create ",
    " def ",
    " edit ",
    " export ",
    " fix ",
    " function",
    " import ",
    " implement ",
    " method",
    " modify ",
    " package",
    " refactor ",
    " remove ",
    " test ",
    " tests ",
    " update ",
)
_PYTHON_CREATE_MARKERS = (
    "create ",
    "create file",
    "implement ",
    "создай файл",
)
_PYTHON_MUTATION_MARKERS = (
    " add ",
    " edit ",
    " fix ",
    " modify ",
    " refactor ",
    " remove ",
    " replace ",
    " update ",
    "добавь",
    "замени",
    "инвертируй",
    "исправь",
    "обнови",
    "переименуй",
    "поменяй",
    "раздели файл",
    "удали",
)
_PYTHON_STRUCTURE_MARKERS = (
    "__main__",
    " class ",
    " dataclass",
    " def ",
    " docstring",
    " function",
    " import ",
    " method",
    " parameter",
    " signature",
    " класс",
    " кавычк",
    " метод",
    " модул",
    " параметр",
    " сигнатур",
    " функци",
    " функция",
    " функции",
)
_PYTEST_PROMPT_MARKERS = (
    " pytest",
    " pytest-",
    " pytest_",
    " tests/",
    " test_",
    "должны пройти",
    "pytest-тест",
    "тесты в ",
)
_LOG_FILTER_PROMPT_MARKERS = (
    " only the lines ",
    " status equals ",
    "только те строки",
    "статус равен",
    "байт-в-байт",
)


@dataclass(frozen=True, slots=True)
class RoutingInput:
    """Normalized task semantics used by routed controllers."""

    prompt: str
    hints: frozenset[str] = field(default_factory=frozenset)

    @property
    def tags(self) -> frozenset[str]:
        """Backward-compatible alias for legacy benchmark terminology."""
        return self.hints


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Combined routing decision for execution loop and tool narrowing."""

    execution_route: ExecutionRoute
    tool_route: ToolRoute

    @property
    def mode_label(self) -> str:
        """Return the human-readable execution label used in logs."""
        if self.execution_route == "deep":
            return "deep/hybrid"
        return "direct"


def build_routing_input(
    prompt: str,
    *,
    hints: Iterable[str] = (),
    tags: Iterable[str] | None = None,
) -> RoutingInput:
    """Build a normalized routing input from prompt text and optional hints."""
    merged_hints = tuple(hints)
    if tags is not None:
        merged_hints = (*merged_hints, *tags)
    return RoutingInput(
        prompt=prompt,
        hints=frozenset(hint.lower() for hint in merged_hints),
    )


def route_task(routing_input: RoutingInput) -> RoutingDecision:
    """Return the complete routing decision for a task-like input."""
    return RoutingDecision(
        execution_route=classify_execution_route(routing_input),
        tool_route=classify_tool_route(routing_input.prompt),
    )


def classify_execution_route(routing_input: RoutingInput) -> ExecutionRoute:
    """Choose the execution loop from task semantics, not task ids."""
    hints = {hint.lower() for hint in routing_input.hints}
    prompt = routing_input.prompt.lower()
    text = f" {prompt} "
    if "filesystem" in hints:
        return "direct"
    if any(marker in prompt for marker in _DIRECT_PROMPT_MARKERS):
        return "direct"
    if "logs" in hints and "filter" in hints:
        return "deep"
    if {"xlsx", "csv", "json"} <= hints:
        return "deep"
    if hints & _DEEP_ROUTE_TAGS:
        return "deep"
    if "edit" in hints and hints & _DEEP_EDIT_FORMAT_TAGS:
        return "deep"
    if (
        "python" in hints
        and hints & _DEEP_PYTHON_EDIT_TAGS
        and not hints & _DIRECT_PYTHON_TASK_TAGS
    ):
        return "deep"

    if ".toml" in prompt or "pyproject.toml" in prompt:
        return "deep"
    if _is_prompt_log_filter_task(text):
        return "deep"
    if _is_prompt_pytest_task(text):
        return "deep"
    tool_route = classify_tool_route(routing_input.prompt)
    if tool_route in {"data", "search", "filesystem"}:
        return "direct"
    if _is_prompt_python_task(text):
        return "deep"
    if any(marker in prompt for marker in _DEEP_PROMPT_MARKERS):
        return "deep"
    if _is_code_file_task(text):
        return "deep"

    return "direct"


def classify_tool_route(prompt: str) -> ToolRoute:
    """Return the generic tool-routing bucket for a prompt."""
    text = f" {prompt.lower()} "
    if _is_filesystem_commit_task(text):
        return "filesystem"
    if _is_search_postprocess_task(text):
        return "search"
    if _is_direct_data_task(text):
        return "data"
    return "hybrid"


def _is_direct_data_task(text: str) -> bool:
    data_markers = (
        ".csv",
        ".db",
        ".ini",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".sqlite",
        ".toml",
        ".tsv",
        ".xlsx",
        ".xml",
        ".yaml",
        ".yml",
        " sqlite",
    )
    operation_markers = (
        "aggregate",
        "bucket",
        "compute",
        "convert",
        "count",
        "deduplicate",
        "dedupe",
        "export",
        "extract",
        "filter",
        "group",
        "groupby",
        "histogram",
        "join",
        "mean",
        "median",
        "pivot",
        "sort",
        "sum",
        "tally",
        "total",
        "агрег",
        "bucket",
        "конверт",
        "объедин",
        "отфильтр",
        "посчитай",
        "подсчитай",
        "преобраз",
        "сгруппир",
        "средн",
        "сумм",
        "экспорт",
    )
    return _has_any(text, data_markers) and _has_any(text, operation_markers)


def _is_search_postprocess_task(text: str) -> bool:
    search_markers = (
        "assert",
        "class name",
        "containing",
        "duplicate",
        "email",
        "extract",
        "find",
        "grep",
        "largest",
        "list files",
        "matches",
        "most lines",
        "todo",
        "встречается подстрока",
        "найди",
        "определения классов",
        "посчитай",
        "собери имена",
    )
    return _has_any(text, search_markers) and (
        _has_any(
            text,
            (
                ".py",
                ".md",
                ".yaml",
                ".yml",
                ".log",
                " files ",
                " under ",
                " каталоге ",
                " файлах ",
            ),
        )
    )


def _is_filesystem_commit_task(text: str) -> bool:
    if _has_any(text, (" rename ", " move ", " delete ", "переименуй", "перемести", "удали")):
        return not _has_any(
            text,
            (
                " class ",
                " docstring",
                " function",
                " line ",
                " method",
                " parameter",
                " строк",
                " класс",
                " метод",
                " параметр",
                " функци",
            ),
        )
    return " convert " in text and _has_any(text, (" package", " directory", " dir "))


def _is_code_file_task(text: str) -> bool:
    return _has_any(text, _CODE_FILE_MARKERS) and _has_any(text, _CODE_TASK_MARKERS)


def _is_prompt_log_filter_task(text: str) -> bool:
    return ".log" in text and _has_any(text, _LOG_FILTER_PROMPT_MARKERS)


def _is_prompt_pytest_task(text: str) -> bool:
    return _has_any(text, _CODE_FILE_MARKERS) and _has_any(text, _PYTEST_PROMPT_MARKERS)


def _is_prompt_python_task(text: str) -> bool:
    if ".py" not in text:
        return False
    if _has_any(text, _PYTHON_CREATE_MARKERS) and _has_any(text, _PYTHON_STRUCTURE_MARKERS):
        return True
    return _has_any(text, _PYTHON_MUTATION_MARKERS)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


__all__ = [
    "ExecutionRoute",
    "RoutingDecision",
    "RoutingInput",
    "ToolRoute",
    "build_routing_input",
    "classify_execution_route",
    "classify_tool_route",
    "route_task",
]
