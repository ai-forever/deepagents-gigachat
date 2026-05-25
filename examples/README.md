# Examples

Runnable examples for using `deepagents-gigachat` with
[`deepagents`](https://github.com/langchain-ai/deepagents).

## Setup

1. Install dependencies from the repository root:

   ```bash
   uv sync
   ```

2. Put GigaChat credentials in `.env` or export them in your shell:

   ```bash
   export GIGACHAT_CREDENTIALS="<your authorization key>"
   # or
   export GIGACHAT_USER="<login>"
   export GIGACHAT_PASSWORD="<password>"
   ```

## Run

From the repository root:

```bash
uv run python examples/basic_agent.py
```

## Example List

| File | Description |
| --- | --- |
| `basic_agent.py` | Minimal agent: create `GigaChat`, wrap it with `create_deep_agent`, and ask one question. |
