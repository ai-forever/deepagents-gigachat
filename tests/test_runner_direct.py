"""Tests for the direct one-action benchmark runner."""

from __future__ import annotations

import json
from typing import Any

from harness_bench import runner_direct
from harness_bench.core import Task, VerifyResult


def test_parse_action_extracts_fenced_json() -> None:
    action = runner_direct._parse_action(  # noqa: SLF001
        '```json\n{"kind":"shell","cmd":"mv old.txt new.txt"}\n```'
    )

    assert action == {"kind": "shell", "cmd": "mv old.txt new.txt"}


def test_parse_action_ignores_trailing_text_after_json() -> None:
    action = runner_direct._parse_action(  # noqa: SLF001
        '{"kind":"python","code":"print(1)"}\n{"ignored": true}'
    )

    assert action == {"kind": "python", "code": "print(1)"}


def test_parse_action_repairs_escaped_single_quotes_in_json_string() -> None:
    action = runner_direct._parse_action(  # noqa: SLF001
        r"""{"kind":"python","code":"with open(\'out.txt\', \'w\') as f:\n    f.write(\'ok\')"}"""
    )

    assert action == {
        "kind": "python",
        "code": "with open('out.txt', 'w') as f:\n    f.write('ok')",
    }


def test_parse_action_accepts_python_literal_style_action() -> None:
    text = (
        r"""{"kind": "python", "code": "with open(\'a\') as src, """
        "\\\n"
        r"""     open(\'b\', \'w\') as dst:\n    pass"}\'"""
    )

    action = runner_direct._parse_action(text)  # noqa: SLF001

    assert action["kind"] == "python"
    assert "with open('a') as src" in action["code"]
    assert "open('b', 'w') as dst" in action["code"]


def test_run_task_direct_executes_model_action(monkeypatch: Any) -> None:
    class FakeModel:
        def invoke(self, _messages: list[dict[str, str]]) -> object:
            class Response:
                content = '{"kind":"python","code":"from pathlib import Path; Path(\\"ok.txt\\").write_text(\\"ok\\\\n\\")"}'

            return Response()

    def fake_build_model(*, model_name: str = runner_direct.DEFAULT_DIRECT_MODEL) -> FakeModel:
        return FakeModel()

    task = Task(
        id="task_direct",
        name="direct",
        prompt="write ok.txt",
        verifier=lambda ws: VerifyResult(
            (ws / "ok.txt").exists() and (ws / "ok.txt").read_text() == "ok\n",
            "ok.txt matches",
        ),
    )

    monkeypatch.setattr(runner_direct, "_build_model", fake_build_model)

    result = runner_direct.run_task_direct(task)

    assert result.passed


def test_run_task_direct_retries_when_named_artifact_missing(monkeypatch: Any) -> None:
    calls = 0

    class FakeModel:
        def invoke(self, _messages: list[dict[str, str]]) -> object:
            nonlocal calls
            calls += 1
            code = "from pathlib import Path; Path('total.txt').write_text('ok\\n')"
            if calls > 1:
                code += "; Path('sum.py').write_text('print(\"ok\")\\n')"

            class Response:
                content = json.dumps({"kind": "python", "code": code})

            return Response()

    def fake_build_model(*, model_name: str = runner_direct.DEFAULT_DIRECT_MODEL) -> FakeModel:
        return FakeModel()

    task = Task(
        id="task_direct_retry",
        name="direct retry",
        prompt="создай файл sum.py и сохрани вывод в файл total.txt",
        verifier=lambda ws: VerifyResult(
            (ws / "sum.py").exists() and (ws / "total.txt").exists(),
            "artifacts exist",
        ),
    )

    monkeypatch.setattr(runner_direct, "_build_model", fake_build_model)

    result = runner_direct.run_task_direct(task, max_actions=2)

    assert result.passed
    assert calls == 2


def test_run_task_direct_retries_after_action_failure(monkeypatch: Any) -> None:
    calls = 0

    class FakeModel:
        def invoke(self, _messages: list[dict[str, str]]) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                code = "import missing_optional_package"
            else:
                code = "from pathlib import Path; Path('ok.txt').write_text('ok\\n')"

            class Response:
                content = json.dumps({"kind": "python", "code": code})

            return Response()

    def fake_build_model(*, model_name: str = runner_direct.DEFAULT_DIRECT_MODEL) -> FakeModel:
        return FakeModel()

    task = Task(
        id="task_direct_action_retry",
        name="direct action retry",
        prompt="write ok.txt",
        verifier=lambda ws: VerifyResult(
            (ws / "ok.txt").exists() and (ws / "ok.txt").read_text() == "ok\n",
            "ok.txt matches",
        ),
    )

    monkeypatch.setattr(runner_direct, "_build_model", fake_build_model)

    result = runner_direct.run_task_direct(
        task,
        max_actions=1,
        action_error_retries=1,
    )

    assert result.passed
    assert calls == 2


def test_run_task_direct_retries_after_python_syntax_error(monkeypatch: Any) -> None:
    calls = 0

    class FakeModel:
        def invoke(self, _messages: list[dict[str, str]]) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                code = "from pathlib import Path\nPath('ok.txt').write_text('ok'\n"
            else:
                code = "from pathlib import Path; Path('ok.txt').write_text('ok\\n')"

            class Response:
                content = json.dumps({"kind": "python", "code": code})

            return Response()

    def fake_build_model(*, model_name: str = runner_direct.DEFAULT_DIRECT_MODEL) -> FakeModel:
        return FakeModel()

    task = Task(
        id="task_direct_syntax_retry",
        name="direct syntax retry",
        prompt="write ok.txt",
        verifier=lambda ws: VerifyResult(
            (ws / "ok.txt").exists() and (ws / "ok.txt").read_text() == "ok\n",
            "ok.txt matches",
        ),
    )

    monkeypatch.setattr(runner_direct, "_build_model", fake_build_model)

    result = runner_direct.run_task_direct(
        task,
        max_actions=1,
        action_error_retries=1,
    )

    assert result.passed
    assert calls == 2


def test_run_task_direct_retries_after_invalid_json(monkeypatch: Any) -> None:
    calls = 0

    class FakeModel:
        def invoke(self, _messages: list[dict[str, str]]) -> object:
            nonlocal calls
            calls += 1

            class Response:
                content = (
                    "not json"
                    if calls == 1
                    else '{"kind":"python","code":"from pathlib import Path; Path(\\"ok.txt\\").write_text(\\"ok\\\\n\\")"}'
                )

            return Response()

    def fake_build_model(*, model_name: str = runner_direct.DEFAULT_DIRECT_MODEL) -> FakeModel:
        return FakeModel()

    task = Task(
        id="task_direct_json_retry",
        name="direct json retry",
        prompt="write ok.txt",
        verifier=lambda ws: VerifyResult(
            (ws / "ok.txt").exists() and (ws / "ok.txt").read_text() == "ok\n",
            "ok.txt matches",
        ),
    )

    monkeypatch.setattr(runner_direct, "_build_model", fake_build_model)

    result = runner_direct.run_task_direct(
        task,
        max_actions=1,
        action_error_retries=1,
    )

    assert result.passed
    assert calls == 2


def test_run_task_direct_retries_after_integer_decimal_format_issue(
    monkeypatch: Any,
) -> None:
    calls = 0

    class FakeModel:
        def invoke(self, _messages: list[dict[str, str]]) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                rows = "sku,qty,price,total\nA,3,100,300.0\n"
            else:
                rows = "sku,qty,price,total\nA,3,100,300\n"
            code = f"from pathlib import Path; Path('invoices.csv').write_text({rows!r})"

            class Response:
                content = json.dumps({"kind": "python", "code": code})

            return Response()

    def fake_build_model(*, model_name: str = runner_direct.DEFAULT_DIRECT_MODEL) -> FakeModel:
        return FakeModel()

    task = Task(
        id="task_direct_format_retry",
        name="direct format retry",
        prompt=(
            "Допиши столбец total, равный произведению qty * price, "
            "и сохрани результат в том же файле invoices.csv."
        ),
        setup_files={"invoices.csv": "sku,qty,price\nA,3,100\n"},
        verifier=lambda ws: VerifyResult(
            (ws / "invoices.csv").read_text() == "sku,qty,price,total\nA,3,100,300\n",
            "invoices.csv matches",
        ),
    )

    monkeypatch.setattr(runner_direct, "_build_model", fake_build_model)

    result = runner_direct.run_task_direct(task, max_actions=2)

    assert result.passed
    assert calls == 2


def test_named_output_artifacts_extracts_outputs_not_inputs() -> None:
    artifacts = runner_direct._named_output_artifacts(  # noqa: SLF001
        "В файле transactions.csv данные. Создай файл sum.py и сохрани вывод в файл total.txt."
    )

    assert artifacts == ["sum.py", "total.txt"]


def test_named_output_artifacts_extracts_same_file_russian_output() -> None:
    artifacts = runner_direct._named_output_artifacts(  # noqa: SLF001
        "Допиши столбец total и сохрани результат в том же файле invoices.csv."
    )

    assert artifacts == ["invoices.csv"]


def test_named_output_artifacts_extracts_make_file_russian_output() -> None:
    artifacts = runner_direct._named_output_artifacts(  # noqa: SLF001
        "Сделай файл rolling.csv с двумя колонками."
    )

    assert artifacts == ["rolling.csv"]


def test_named_output_artifacts_extracts_into_one_file_russian_output() -> None:
    artifacts = runner_direct._named_output_artifacts(  # noqa: SLF001
        "Объедини их в один файл merged.csv, удалив одинаковые строки."
    )

    assert artifacts == ["merged.csv"]


def test_format_issue_detail_detects_integer_decimal_suffixes(tmp_path: Any) -> None:
    (tmp_path / "invoices.csv").write_text("sku,total\nA,300.0\n")

    detail = runner_direct._format_issue_detail(  # noqa: SLF001
        "Добавь столбец total и сохрани результат в том же файле invoices.csv.",
        tmp_path,
    )

    assert detail is not None
    assert "integer-like decimal suffixes" in detail


def test_format_issue_detail_detects_apache_status_filter_mismatch(
    tmp_path: Any,
) -> None:
    (tmp_path / "access.log").write_text(
        '10.0.0.1 - - [13/May/2026:10:00:00 +0000] "GET / HTTP/1.1" 200 100\n'
        '10.0.0.2 - - [13/May/2026:10:00:01 +0000] "GET /a HTTP/1.1" 404 50\n'
    )
    (tmp_path / "not_found.log").write_text("")

    detail = runner_direct._format_issue_detail(  # noqa: SLF001
        "Сохрани в файл not_found.log только строки, у которых HTTP-статус равен 404.",
        tmp_path,
    )

    assert detail is not None
    assert "Apache log status filter issue" in detail
    assert "404" in detail


def test_controller_prompt_uses_route_hint() -> None:
    messages = runner_direct._controller_messages(  # noqa: SLF001
        "Find the .py file under project/ with the most lines"
    )

    assert "search/extraction" in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_controller_prompt_includes_generic_direct_safeguards() -> None:
    messages = runner_direct._controller_messages(  # noqa: SLF001
        "Count words in input.txt"
    )

    prompt = messages[0]["content"]
    assert "optional writer packages" in prompt
    assert "write exactly N non-empty lines" in prompt
    assert "Do not replace a non-empty source file with an empty destination file" in prompt
    assert "header-aware parser" in prompt
    assert "csv.DictReader/DictWriter" in prompt
    assert "preserve the apparent input scalar types" in prompt
    assert "statistics.median" in prompt
    assert "sorted_values[len(sorted_values)//2]" in prompt
    assert "deterministic row and column ordering" in prompt
    assert "no index column" in prompt
    assert "no blank rows" in prompt
    assert "preserve sheet/header order and cell scalar types" in prompt
    assert "do not use naive line.split()" in prompt
    assert "integer immediately after the closing quote" in prompt
    assert "extract HH from timestamps" in prompt
    assert "log filtering by HTTP status" in prompt
    assert "scan every regular file" in prompt
    assert "exact single-character occurrences" in prompt
    assert "whitespace-separated tokens" in prompt


def test_controller_messages_include_previous_failure_feedback() -> None:
    messages = runner_direct._controller_messages(  # noqa: SLF001
        "write ok.txt",
        previous_failure="Action exited with 1",
    )

    assert "previous action attempt failed" in messages[1]["content"]
    assert "Action exited with 1" in messages[1]["content"]


def test_preflight_action_reports_python_syntax_details() -> None:
    try:
        runner_direct._preflight_action(  # noqa: SLF001
            {"kind": "python", "code": "if True print('bad')"}
        )
    except ValueError as exc:
        detail = str(exc)
    else:  # pragma: no cover - defensive assertion for a helper test
        raise AssertionError("expected syntax error")

    assert "Python action has SyntaxError" in detail
    assert "line 1" in detail


def test_preflight_action_adds_with_statement_hint() -> None:
    try:
        runner_direct._preflight_action(  # noqa: SLF001
            {"kind": "python", "code": "with open('a') as a,\n    open('b') as b:\n    pass"}
        )
    except ValueError as exc:
        detail = str(exc)
    else:  # pragma: no cover - defensive assertion for a helper test
        raise AssertionError("expected syntax error")

    assert "multiple context managers" in detail
    assert "bare trailing comma" in detail


def test_preflight_action_adds_regex_quote_hint() -> None:
    try:
        runner_direct._preflight_action(  # noqa: SLF001
            {
                "kind": "python",
                "code": "import re\nre.search(r'^return 'help'$', text)",
            }
        )
    except ValueError as exc:
        detail = str(exc)
    else:  # pragma: no cover - defensive assertion for a helper test
        raise AssertionError("expected syntax error")

    assert "regex pattern contains quote characters" in detail
