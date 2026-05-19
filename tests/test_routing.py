"""Tests for shared routing policy outside benchmark adapters."""

from __future__ import annotations

from deepagents_gigachat.routing import (
    build_routing_input,
    classify_execution_route,
    classify_tool_route,
    route_task,
)


def test_classify_tool_route_is_generic() -> None:
    assert classify_tool_route("Inner-join users.csv and orders.csv by user_id") == "data"
    assert classify_tool_route("Find the .py file under project/ with the most lines") == "search"
    assert classify_tool_route("Move src/old/* to src/new/") == "filesystem"
    assert classify_tool_route("Implement Stack so pytest passes") == "hybrid"


def test_classify_execution_route_sends_code_semantics_to_deep() -> None:
    assert classify_execution_route(build_routing_input("do it", hints=("python", "impl"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("python", "pytest"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("python", "fix"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("python", "refactor"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("toml", "edit"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("python", "create"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("python", "edit"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("logs", "filter"))) == "deep"
    assert classify_execution_route(build_routing_input("do it", hints=("xlsx", "csv", "json"))) == "deep"
    assert (
        classify_execution_route(
            build_routing_input("Bump version in pyproject.toml", hints=("config", "edit"))
        )
        == "deep"
    )
    assert classify_execution_route(build_routing_input("Create utils.py with function add(a, b).")) == "deep"
    assert (
        classify_execution_route(
            build_routing_input(
                "Создай файл greeting.py с функцией greet(name: str) -> str."
                " Она должна возвращать строку формата 'Привет, <name>!'."
            )
        )
        == "deep"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "В файле version.py обнови значение переменной VERSION"
                " с '1.0.0' на '1.0.1'."
            )
        )
        == "deep"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "Создай файл linked_list.py с классом LinkedList."
                " Тесты в tests/test_ll.py должны пройти."
            )
        )
        == "deep"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "В файле utils.py три функции: keep_one, unused_helper, keep_two."
                " Удали определение функции unused_helper целиком."
            )
        )
        == "deep"
    )


def test_classify_execution_route_sends_data_and_search_to_direct() -> None:
    assert classify_execution_route(build_routing_input("do it", hints=("csv", "compute"))) == "direct"
    assert classify_execution_route(build_routing_input("do it", hints=("grep", "search"))) == "direct"
    assert classify_execution_route(build_routing_input("do it", hints=("filesystem", "edit"))) == "direct"
    assert (
        classify_execution_route(build_routing_input("do it", hints=("filesystem", "edit", "refactor")))
        == "direct"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "Перенеси функцию helper() из a.py в b.py",
                hints=("python", "refactor"),
            )
        )
        == "direct"
    )
    assert (
        classify_execution_route(build_routing_input("do it", hints=("python", "execute", "compute")))
        == "direct"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "В каталоге lib лежат 6 .py-файлов. Найди в них все определения"
                " классов и собери имена классов в файл classes.txt."
            )
        )
        == "direct"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "В файле access.log около 48 строк. Сгруппируй запросы по часу"
                " и сохрани результат в hourly.csv с заголовком 'hour,count'."
            )
        )
        == "direct"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "Переименуй каталог src/old в src/new — все файлы alpha.py и"
                " beta.py из него должны оказаться в src/new."
            )
        )
        == "direct"
    )
    assert (
        classify_execution_route(
            build_routing_input(
                "В файле script.py две строки на верхнем уровне."
                " Оберни их в функцию main() и добавь блок"
                " if __name__ == '__main__': main()."
            )
        )
        == "direct"
    )


def test_classify_execution_route_handles_log_filter_as_deep() -> None:
    assert (
        classify_execution_route(
            build_routing_input(
                "В файле access.log сохрани в not_found.log только те строки,"
                " у которых HTTP-статус равен 404, байт-в-байт."
            )
        )
        == "deep"
    )


def test_route_task_combines_execution_and_tool_routes() -> None:
    decision = route_task(
        build_routing_input(
            "Find the .py file under project/ with the most lines",
            hints=("grep", "search"),
        )
    )

    assert decision.execution_route == "direct"
    assert decision.tool_route == "search"
    assert decision.mode_label == "direct"
