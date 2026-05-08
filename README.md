# deepagents-gigachat

A [`HarnessProfile`](https://docs.langchain.com/oss/python/deepagents/profiles#harness-profiles)
for [`deepagents`](https://github.com/langchain-ai/deepagents) tuned for
[GigaChat](https://giga.chat/) models.

The profile replaces the default `deepagents` system prompt, rewrites the
descriptions of file and shell tools (`ls`, `read_file`, `write_file`, `glob`,
`grep`, `edit_file`, `execute`) to match GigaChat's tool-calling behavior, and
adds a `think` middleware tool for structured intermediate reasoning.

Once installed, the profile is registered automatically via the
`deepagents.harness_profiles` entry point — no code changes required.

## Structure

- `deepagents_gigachat/harness_profile.py` — GigaChat `HarnessProfile` implementation
- `deepagents_gigachat/prompts.py` — base prompt used by the profile
- `deepagents_gigachat/__init__.py` — public entry point exporting `register_harness()`

## Requirements

- Python 3.12+
- `uv` (for dependency installation and execution)

## Installation

```bash
uv sync
```

Once published to PyPI, downstream users can install with:

```bash
pip install deepagents-gigachat
```

## Configuration

Provide one of the authentication options in your shell environment. If your
launcher loads dotenv files, for example `deepagents-cli`, these values can also
live in `.env`:

- `GIGACHAT_CREDENTIALS`
- or `GIGACHAT_USER` + `GIGACHAT_PASSWORD`

Optional GigaChat settings:

```bash
GIGACHAT_BASE_URL="https://gigachat.sberdevices.ru/v1"
GIGACHAT_MODEL="GigaChat-3-Ultra"
GIGACHAT_VERIFY_SSL_CERTS=False
GIGACHAT_PROFANITY_CHECK=False
```

## Use With `deepagents`

Install this package into the same Python environment where `deepagents` runs:

```bash
pip install deepagents-gigachat
```

For local development, install the built wheel instead:

```bash
uv build
uv pip install dist/*.whl
```

After installation, `deepagents` discovers the profile automatically through the
`deepagents.harness_profiles` entry point:

```toml
[project.entry-points."deepagents.harness_profiles"]
gigachat = "deepagents_gigachat:register_harness"
```

The package entry point is named `gigachat` for discovery. The harness profile
is registered under both provider keys: `gigachat` for model specs such as
`gigachat:GigaChat-3-Ultra`, and `giga` as a compatibility alias.

## Use With `deepagents-cli`

Install the CLI and GigaChat provider into the same environment:

```bash
uv pip install deepagents-cli langchain-gigachat deepagents-gigachat
```

Then configure `~/.deepagents/config.toml`:

```toml
[models]
default = "gigachat:GigaChat-3-Ultra"

[models.providers.gigachat]
models = [
    "GigaChat-3-Ultra",
    "GigaChat-2-Max",
    "GigaChat-Max",
    "GigaChat-Pro",
    "GigaChat",
]
class_path = "langchain_gigachat.chat_models.gigachat:GigaChat"
api_key_env = "GIGACHAT_CREDENTIALS"

[models.providers.gigachat.params]
base_url = "https://gigachat.sberdevices.ru/v1"
verify_ssl_certs = false
profanity_check = false
timeout = 600

[models.providers.gigachat.profile]
tool_calling = true
default_model_hint = "GigaChat-3-Ultra"
```

If you use `GIGACHAT_USER` and `GIGACHAT_PASSWORD` instead of
`GIGACHAT_CREDENTIALS`, remove `api_key_env` from the config and keep the user
and password values in `.env` or shell environment variables.

Run the CLI:

```bash
deepagents
```

## Lint

Linting, tests, and package build checks are required in CI:

```bash
uv run ruff check .
uv run pytest
uv build
```

