# Deep Agents на GigaChat — практический блокнот

[Deep Agents](https://github.com/langchain-ai/deepagents) — агентный харнесс от LangChain
поверх LangGraph: агент получает файловую систему, shell, планировщик, субагентов и навыки
«из коробки». Блокнот показывает его на **GigaChat** — сквозной пример агента-аналитика,
который читает данные, пишет и запускает код и оформляет отчёт.

Профиль харнесса [`deepagents-gigachat`](https://github.com/ai-forever/deepagents-gigachat)
подключается автоматически (через entry point `deepagents.harness_profiles`) и подстраивает
харнесс под особенности GigaChat — достаточно, чтобы пакет был установлен.

## Требования

- **Python 3.12+**
- [`uv`](https://docs.astral.sh/uv/) для окружения и запуска

## Установка окружения

Всё ставится в изолированный `.venv` (базовый Python не трогаем):

```bash
uv sync
```

`uv sync` создаст `.venv` с Python 3.12+ и поставит зависимости из `pyproject.toml`
(зафиксированы в `uv.lock`).

## Ключи GigaChat

```bash
cp .env.example .env
```

Заполните `.env`: авторизацию (`GIGACHAT_CREDENTIALS` — один base64-ключ,
либо `GIGACHAT_USER` + `GIGACHAT_PASSWORD`) и модель (`GIGACHAT_MODEL`).
Настоящий `.env` не коммитится (см. `.gitignore`).

## Наблюдаемость

**Arize Phoenix** держим отдельным процессом, чтобы рестарт ядра его не ронял.
В отдельном терминале из папки блокнота:

```bash
uv run phoenix serve
```

UI и коллектор — на `http://localhost:6006` (данные копятся в `~/.phoenix`).
Ячейка наблюдаемости в блокноте только подключает инструментирование
(`register(endpoint=...)`), сервер не поднимает.

## Запуск

```bash
uv run jupyter lab
```

Откройте `deepagents_gigachat_practice.ipynb` и выполняйте ячейки сверху вниз.
В VS Code выберите интерпретатор/ядро из `.venv`.

## Что внутри

1. Окружение (эта инструкция)
2. Конфигурация GigaChat из `.env`
3. Наблюдаемость: подключение к Phoenix (OpenInference)
4. Разминка: файловая система без диска (`StateBackend`)
5. Аналитик + shell на реальных данных (`LocalShellBackend`, `virtual_mode=True`)
6. Бонус: навыки (skills) с прогрессивным раскрытием
