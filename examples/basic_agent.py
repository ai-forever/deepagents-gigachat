"""Минимальный пример: deep-агент на GigaChat.

Что происходит:
1. Регистрируем харнесс-профиль для GigaChat через
   `deepagents_gigachat.register_harness()`. После этого `deepagents`
   будет использовать наш системный промпт, переопределённые описания
   инструментов (`read_file`, `write_file`, `grep`, `execute` и т.д.)
   и добавит инструмент `think`.
2. Создаём модель `GigaChat` через `langchain-gigachat`.
3. Оборачиваем её в `create_deep_agent` из `deepagents` — профиль
   подцепится автоматически по провайдеру модели (`giga`).

Примечание про регистрацию: пакет `deepagents-gigachat` объявляет
entry point `deepagents.harness_profiles`, поэтому `register_harness()`
будет вызвана `deepagents` сама при первом обращении. Здесь мы вызываем
её явно, чтобы зависимость пакета была видна прямо в коде примера.
Повторный вызов безопасен.

Запуск из корня репозитория:

    uv run python examples/basic_agent.py
"""

from __future__ import annotations

import os
from pathlib import Path

from deepagents import create_deep_agent
from langchain_gigachat import GigaChat

from deepagents_gigachat import register_harness

register_harness()

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_env() -> None:
    """Подгрузить переменные из .env в корне репозитория, если есть dotenv."""
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def build_agent() -> object:
    """Собрать GigaChat и обернуть его в deep-агента."""
    model = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat-3-Ultra"),
        base_url=os.getenv("GIGACHAT_BASE_URL", "https://gigachat.sberdevices.ru/v1"),
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=600,
    )
    return create_deep_agent(model=model)


def main() -> None:
    _load_env()

    if not os.getenv("GIGACHAT_CREDENTIALS") and not (
        os.getenv("GIGACHAT_USER") and os.getenv("GIGACHAT_PASSWORD")
    ):
        raise SystemExit(
            "Не заданы учётные данные GigaChat. "
            "Укажи GIGACHAT_CREDENTIALS либо пару GIGACHAT_USER + GIGACHAT_PASSWORD."
        )

    agent = build_agent()

    question = "Привет! Назови три факта про язык Python простыми словами."
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    last_message = result["messages"][-1]
    print("=" * 60)
    print("Вопрос:", question)
    print("-" * 60)
    print("Ответ:", last_message.content)
    print("=" * 60)


if __name__ == "__main__":
    main()
