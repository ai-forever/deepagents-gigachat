# deepagents-gigachat

![harness-bench-fast: +13.4 pp with the GigaChat profile](docs/assets/benchmark_profile_uplift.png)

A [`HarnessProfile`](https://docs.langchain.com/oss/python/deepagents/profiles#harness-profiles)
for [`deepagents`](https://github.com/langchain-ai/deepagents) tuned for
[GigaChat](https://giga.chat/) models. On the
[`harness-bench-fast`](https://github.com/ai-forever/harness-bench-fast)
231-task agent benchmark the profile lifts `GigaChat-3-Ultra` from
**164 / 231 (71.0 %)** to **195 / 231 (84.4 %)** — see the
[Benchmark](#benchmark) section for the methodology and full results.

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

Downstream users can install the published package from PyPI with:

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

Step-by-step setup for using GigaChat as the default model in the
`deepagents` CLI through its config file.

### 1. Install the CLI, the GigaChat provider, and this plugin

All three must end up in the **same** Python environment so that the CLI
can both construct a `GigaChat` model and discover the harness profile
via the `deepagents.harness_profiles` entry point:

```bash
uv pip install deepagents-cli langchain-gigachat deepagents-gigachat
```

(or `pip install ...` if you're not using `uv`).

### 2. Provide credentials

GigaChat accepts two authentication styles. Pick one.

**Option A: Authorization Key (one base64-encoded string).** Get the key
from `developers.sber.ru` → your project → credentials section, then
export it:

```bash
export GIGACHAT_CREDENTIALS="<base64-encoded auth key>"
```

**Option B: User + password.** If you have a `user`/`password` pair
instead of a single key:

```bash
export GIGACHAT_USER="<your client id>"
export GIGACHAT_PASSWORD="<your client secret>"
```

You can also put either pair into a `.env` file next to where you launch
the CLI — `deepagents` reads `.env` on startup. The plugin itself never
parses these variables: `langchain-gigachat` picks them up when it
constructs the model.

### 3. Configure `~/.deepagents/config.toml`

Create the file (the directory may not exist yet — `mkdir -p ~/.deepagents`
first) and put the snippet below into it. Each block is annotated.

```toml
[models]
# The model used when you launch `deepagents` with no extra flags.
# Format: "<provider>:<model name>". The provider key here ("gigachat")
# is the same one this plugin registers its harness profile under.
default = "gigachat:GigaChat-3-Ultra"

[models.providers.gigachat]
# Models exposed to the CLI's "/model" picker. Add or remove freely.
models = [
    "GigaChat-3-Ultra",
    "GigaChat-2-Max",
    "GigaChat-Max",
    "GigaChat-Pro",
    "GigaChat",
]
# Tells the CLI which Python class to instantiate when a `gigachat:*`
# spec is requested.
class_path = "langchain_gigachat.chat_models.gigachat:GigaChat"
# If you authenticate via GIGACHAT_CREDENTIALS, this line wires it up.
# Remove this line if you use GIGACHAT_USER + GIGACHAT_PASSWORD instead.
api_key_env = "GIGACHAT_CREDENTIALS"

[models.providers.gigachat.params]
# Constructor kwargs passed straight to `GigaChat(...)`. Anything that
# `langchain_gigachat.GigaChat` accepts can go here.
base_url = "https://gigachat.sberdevices.ru/v1"
verify_ssl_certs = false
profanity_check = false
timeout = 600
# Optional sampling knobs (defaults are sensible; uncomment to override):
# temperature = 0.0
# top_p = 1.0
# repetition_penalty = 1.0

[models.providers.gigachat.profile]
# Tells the CLI's profile resolver that this provider supports tool
# calling and which model to default to when the user types just
# "gigachat" without a model name.
tool_calling = true
default_model_hint = "GigaChat-3-Ultra"
```

### 4. Run the CLI

```bash
deepagents
```

On startup the CLI loads the config, instantiates `GigaChat` with the
parameters above, and `deepagents` automatically picks up this plugin's
harness profile via its `deepagents.harness_profiles` entry point — so
GigaChat-specific system prompt, tool description overrides and the
`think` middleware are applied without any extra code.

### Switching models

Three independent ways to override the default at runtime:

- **Inside the CLI:** type `/model gigachat:GigaChat-Pro` to switch the
  current session.
- **From the shell, per-launch:** `deepagents --model gigachat:GigaChat-Max`.
- **From the environment:** set `GIGACHAT_MODEL=GigaChat-Pro` before
  launching. (This is honoured by `langchain-gigachat` itself when the
  model name isn't pinned in the config.)

### Self-hosted / IFT GigaChat endpoint

Point `base_url` at your custom host. For Sber's internal IFT, for
example:

```toml
[models.providers.gigachat.params]
base_url = "https://gigachat.ift.sberdevices.ru/v1"
```

Everything else stays the same.

## Examples

Runnable examples live in [`examples/`](examples/). The simplest one is
`examples/basic_agent.py`: it constructs a `GigaChat` model, wraps it in
`create_deep_agent`, and asks a single question. Run it with:

```bash
uv run python examples/basic_agent.py
```

## Benchmark

The benchmark used to validate every profile version of this plugin
now lives in its own repo:
[`ai-forever/harness-bench-fast`](https://github.com/ai-forever/harness-bench-fast).
It is a self-contained 231-task agent evaluation covering file
creation/editing, refactors, project-wide `grep`/`glob`, CSV / JSON /
JSONL / YAML / TOML / INI / XLSX / SQLite pipelines, pytest-graded
implementations, composite multi-step pipelines, and `MEMORY.md`
discipline. Every verifier is mechanical — no LLM-as-judge.

Latest contribution of this plugin on that bench, against
`GigaChat-3-Ultra` at `gigachat.sberdevices.ru/v1`:

| Configuration                          | PASS / 231 | %      | Δ                  |
| -------------------------------------- | ---------- | ------ | ------------------ |
| stock `deepagents`, no profile         | 164 / 231  | 71.0 % | —                  |
| `deepagents` + this plugin (v9)        | 195 / 231  | 84.4 % | +31 (+13.4 pp)     |

v9 = `ThinkToolMiddleware` + `ShellSafetyMiddleware` +
`ToolContractMiddleware` + `LoopBreakerMiddleware` + the
`base_system_prompt` and tool description overrides.

## Lint

Linting, tests, and package build checks are required in CI:

```bash
uv run ruff check .
uv run pytest
uv build
```
