# Примеры

Здесь лежат запускаемые примеры использования `deepagents-gigachat`
вместе с библиотекой [`deepagents`](https://github.com/langchain-ai/deepagents).

## Подготовка

1. Установи зависимости в корне репозитория:

   ```bash
   uv sync
   ```

2. Положи учётные данные GigaChat в `.env` или экспортируй в shell:

   ```bash
   export GIGACHAT_CREDENTIALS="<твой ключ авторизации>"
   # либо
   export GIGACHAT_USER="<логин>"
   export GIGACHAT_PASSWORD="<пароль>"
   ```

## Запуск

Из корня репозитория:

```bash
uv run python examples/basic_agent.py
```

## Список примеров

| Файл | О чём |
| --- | --- |
| `basic_agent.py` | Минимальный агент: создаём `GigaChat`, оборачиваем в `create_deep_agent`, задаём один вопрос. |
