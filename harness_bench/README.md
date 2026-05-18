# harness_bench

A small in-tree benchmark of **200 file-operation tasks** used to drive
`deepagents` + `langchain-gigachat` (with this repository's
[`HarnessProfile`](../deepagents_gigachat/harness_profile.py)) and see how
well the model handles common operations: file creation, code edits,
renames, search, format conversion, multi-file refactoring, running
pytest, etc.

## What's inside

| File | Purpose |
| --- | --- |
| `tasks.py` | Tasks 1–30 plus the top-level `ALL_TASKS` registry. |
| `tasks_extra.py` | Tasks 31–60: second wave (multi-file refactors, dedupe, log filtering, CSV ↔ markdown conversions). |
| `tasks_more.py` | Tasks 61–100: third wave (.env edits, nested JSON, dataclasses, simple regex extraction, INI/TOML/YAML stubs, CSV row splitting). |
| `tasks_hard.py` | Tasks 101–150: harder wave (CSV/Excel/SQLite aggregates, JSON/JSONL, YAML/INI/TOML, Python implementation + pytest, multi-file `grep`, Apache log parsing). |
| `tasks_extreme.py` | Tasks 151–200: hardest wave (composite pipelines, archives, project-wide refactors, algorithms with pytest, statistics, XML/markdown, three-way joins). |
| `verifiers.py` | Helpers for writing verifiers: `file_exists`, `file_contains`, `file_lines_equal`, `file_matches_regex`, `json_file_has`, `python_runs`, `python_callable_returns`, `pytest_passes`, `xlsx_cell_equals`, `sqlite_query_returns`, `all_of`, etc. |
| `core.py` | `Task` (dataclass) and `VerifyResult`. Supports `setup_callback`/`gold_callback` hooks for binary fixtures (xlsx, sqlite, zip, tar). |
| `runner.py` | Runs a task inside an isolated temporary directory backed by `LocalShellBackend(virtual_mode=True)`, with optional `--concurrency` via a thread pool. |
| `runner_cli.py` | Alternative runner that drives an external CLI agent (default: `free-code -p --model haiku --dangerously-skip-permissions`). |
| `runner_openrouter.py` | Runner for any OpenAI-compatible OpenRouter model via `langchain-openai`. |
| `__main__.py` | CLI: `list`, `run`, `run-cli`, `run-openrouter`, `verify-gold`. |

Each task is independent: the runner creates a fresh
`tempfile.TemporaryDirectory`, writes `setup_files` (and optionally calls
`setup_callback` for binary fixtures), then points `LocalShellBackend` at
that directory as its `root_dir`. The agent only sees those files —
`virtual_mode=True` blocks path traversal through the file tools, though
`execute` still spawns a real shell on the host (the benchmark is meant
for a trusted local environment). After the agent stops, the per-task
verifier inspects the workspace.

## Task categories (200 in total)

- **File creation** (incl. 1–5, 29, 44, 46, 89, 99): `hello.py`,
  `data.json`, `src/utils.py`, `numbers.txt`, `greeting.py`, `.gitignore`,
  `requirements.txt`, `src/__init__.py`, `.pre-commit-config.yaml`,
  `README.md`.
- **Surgical code edits** (incl. 6–10, 16–18, 53, 57, 67, 78, 83, 87,
  90): toggling `DEBUG`, renaming a function, bumping a version,
  replacing a string, bumping `pyproject.toml`, adding type hints,
  `from __future__ import annotations`, adding a docstring, replacing
  quotes, flipping booleans, sorting imports.
- **Reading and counting** (incl. 11–15, 25, 32–33, 35, 38–39, 41, 43,
  47, 51–52, 71, 86, 92, 98, 110, 175–182): counting `.py` files, picking
  out `TODO` lines, line counts, sums, sorts, dedupe, percentiles,
  rolling averages, histograms, pivot tables, z-score outliers.
- **Refactoring** (incl. 19, 20, 24, 31, 42, 45, 50, 56, 74, 80, 94,
  162–166, 192): removing a deprecated function, moving a function
  between files, headers, multi-file renames, extracting constants into
  a module, splitting a module by class, converting to `@dataclass`,
  project-wide import rewrites.
- **Filesystem operations** (incl. 21–23, 30, 48, 60, 159–161, 198):
  rename, delete, append, copyright headers, gzip, zip create/extract,
  tar extract, rename a directory.
- **JSON / config** (incl. 26–28, 49, 54–55, 61–63, 68–70, 91, 115–122,
  185, 197): adding a key, bumping a dependency, CSV ↔ JSON, CSV ↔
  TSV, conftest fixtures, swapping CSV columns, nested JSON edits,
  YAML/INI/TOML edits, YAML front-matter parsing.
- **Python implementation + pytest** (incl. 125–134, 167–174,
  193–195, 200): `fib`, `factorial`, `is_palindrome`, `count_vowels`,
  `quicksort`, `binary_search`, `is_balanced`, `LRUCache`, `LinkedList`,
  `TreeNode + inorder`, `is_anagram`, `two_sum`, `memoize`, `Timer`
  context manager, `MyRange` iterator, `PriorityQueue`.
- **Multi-file `grep`/`glob` search** (incl. 135–144, 186–187):
  counting `import`/`def`/`assert` across a project tree, listing
  files containing a marker, finding duplicates, dead-function
  detection.
- **Excel (xlsx)** (incl. 111–113, 148, 158, 196): cell extraction,
  column sums, cell updates, CSV ↔ xlsx, per-sheet split.
- **SQLite** (incl. 123–124, 149, 191, 199): counts, sums, JOIN +
  CSV export, JSON export, filtered queries.
- **Apache log parsing** (incl. 145–147, 189): top IP, 5xx count,
  status filter, hourly aggregation.
- **Composite pipelines** (incl. 151–158, 188–190): CSV → JSON
  aggregates with filter+groupby+sort, SQLite JOIN → JSON, xlsx →
  markdown report, three-way joins, multi-CSV concat + dedupe.

Every verifier is mechanical — no LLM-as-judge: exact content checks,
regex matches, line lists, JSON parsing, running `python file.py` and
comparing stdout, or importing a module and calling a function.

## Running the benchmark

The bench lives inside the same `uv` project, so no separate install is
needed — just sync deps:

```bash
uv sync
```

Then provide GigaChat credentials (same as `examples/basic_agent.py`):

```bash
export GIGACHAT_CREDENTIALS=...
# or
export GIGACHAT_USER=...
export GIGACHAT_PASSWORD=...
```

These can also live in `.env` — `runner.py` loads it from the repo root.

### List all tasks

```bash
uv run python -m harness_bench list
```

### Run the benchmark

```bash
# all 200 tasks in sequence
uv run python -m harness_bench run

# parallel run (5 tasks at a time — each in its own workspace)
uv run python -m harness_bench run --concurrency 5

# specific tasks only
uv run python -m harness_bench run \
    --task task_01_create_hello \
    --task task_06_toggle_debug

# keep the temp workspaces on disk (useful for debugging failures)
uv run python -m harness_bench run --task task_20_move_function --keep
```

At the end the runner prints `Passed: N/200` and a one-line summary for
every failed task.

### Run against another model

Through an external CLI agent (default driver: `free-code` + Claude
Haiku):

```bash
uv run python -m harness_bench run-cli --concurrency 5
uv run python -m harness_bench run-cli \
    --cli-command 'free-code -p --model opus --dangerously-skip-permissions'
```

Through any OpenAI-compatible OpenRouter model:

```bash
export OPENROUTER_API_KEY=...
uv run python -m harness_bench run-openrouter \
    --model anthropic/claude-haiku-4.5 --concurrency 5
```

`run-openrouter` does **not** apply the GigaChat harness profile — it
exercises stock `deepagents` against the chosen model.

### Verifying without an LLM

Each task carries `gold_files` — the workspace state a "perfect" agent
would produce. The `verify-gold` command applies the gold solution and
runs the verifier without ever calling the model. Handy when adding new
tasks to make sure the verifier accepts a correct solution:

```bash
uv run python -m harness_bench verify-gold
```

## Results

All runs use `--concurrency 5`. The `deepagents` rows use this repo
(`uv run python -m harness_bench run` for GigaChat, `run-openrouter` for
OpenRouter models). The `free-code` rows use Claude Code CLI v2.1.119.

| Date | Runner | Model | Harness adapt | Result | % |
| --- | --- | --- | --- | --- | --- |
| 2026-05-14 | `deepagents` | Mistral Small 3.2 24B Instruct | no | 94 / 200 | 47.0 % |
| 2026-05-13 | `deepagents` | Llama 3.3 70B Instruct | no | 100 / 200 | 50.0 % |
| 2026-05-15 | `pi-mono` | Llama 3.3 70B Instruct | yes (built-in) | 127 / 200 | 63.5 % |
| 2026-05-14 | `deepagents` | GPT-4.1-nano | no | 115 / 200 | 57.5 % |
| 2026-05-14 | `deepagents` | GPT-3.5-turbo | no | 119 / 200 | 59.5 % |
| 2026-05-13 | `deepagents` | GigaChat-3-Ultra | no | 134 / 200 | 67.0 % |
| 2026-05-14 | `deepagents` | GigaChat-3-Pro | yes (v3) | 137 / 200 | 68.5 % |
| 2026-05-14 | `pi-mono` | GPT-4.1-nano | ? (run by colleague) | 141 / 200 | 70.5 % |
| 2026-05-14 | `deepagents` | Qwen3-Coder-30B-A3B Instruct | no | 163 / 200 | 81.5 % |
| 2026-05-14 | `deepagents` | GigaChat-3-Ultra | yes (v3) | 164 / 200 | 82.0 % |
| 2026-05-15 | `deepagents` | GigaChat-2-Max | yes (v3) | 165 / 200 | 82.5 % |
| 2026-05-18 | `deepagents` | **GigaChat-3-Ultra** (IFT) | **yes (v4)** | **169 / 200** | **84.5 %** |
| 2026-05-14 | `deepagents` | DeepSeek V4 Flash | no | 165 / 200 | 82.5 % |
| 2026-05-14 | `pi-mono` | GPT-4o-mini | ? (run by colleague) | 166 / 200 | 83.0 % |
| 2026-05-13 | `deepagents` | GPT-4.1-mini | no | 168 / 200 | 84.0 % |
| 2026-05-18 | `pi-mono` | **GigaChat-3-Ultra** (IFT, token via /v1/token) | yes (ext: gigachat 0.1.1) | **170 / 200** | **85.0 %** |
| 2026-05-14 | `deepagents` | Qwen3.5-397B-A17B | no | 172 / 200 | 86.0 % |
| 2026-05-14 | `deepagents` | GLM-4.6 | no | 174 / 200 | 87.0 % |
| 2026-05-13 | `deepagents` | Claude Haiku 4.5 | yes (built-in) | 177 / 200 | 88.5 % |
| 2026-05-14 | `pi-mono` | GPT-4.1-mini | ? (run by colleague) | 179 / 200 | 89.5 % |
| 2026-05-14 | `deepagents` | GLM-5.1 | no | 180 / 200 | 90.0 % |
| 2026-05-14 | `deepagents` | Claude Sonnet 4.5 | no | 185 / 200 | 92.5 % |
| 2026-05-13 | `free-code` | Claude Haiku 4.5 | yes (built-in) | 185 / 200 | 92.5 % |
| 2026-05-14 | `deepagents` | Claude Opus 4.7 | yes (built-in) | 188 / 200 | 94.0 % |
| 2026-05-15 | `pi-mono` | Claude Haiku 4.5 | yes (built-in) | 190 / 200 | 95.0 % |
| 2026-05-13 | `free-code` | **Claude Opus 4.7** | yes (built-in) | **195 / 200** | **97.5 %** |

The GigaChat-3-Ultra row with `yes (v4)` is the current pinned
configuration of this repository — the harness profile registered by
`deepagents_gigachat` (expanded prompt with explicit guidance on
required-output strictness, turn-budget control, and `think`-tool
discipline) plus a wider transient retry budget (`max_retries=20` in
`runner.py`, which lets the agent ride out IFT IP-throttle bursts of
500 / 403 without dropping tasks). `v3` is the previous prompt; without
either, GigaChat-3-Ultra scores 134 / 200 on the same bench. Raw run
logs are written to `harness_bench/runs/`.

The `yes (built-in)` rows pick up harness profiles that ship inside
`deepagents` itself (currently only `anthropic:claude-opus-4-7`,
`anthropic:claude-sonnet-4-6`, `anthropic:claude-haiku-4-5` and a few
`openai:gpt-5.x-codex` keys). Other GPT-* / Sonnet 4.5 / open-weights
rows fall back to the generic provider profile and score with no
model-specific adapt.

## Adding a task

1. In one of the task modules (`tasks.py`, `tasks_extra.py`,
   `tasks_more.py`, `tasks_hard.py`, `tasks_extreme.py` — pick the one
   that fits the wave / difficulty) describe a `Task(...)` — id, prompt,
   `setup_files`, `gold_files`, `verifier`.
2. Wire it into the corresponding module's `*_TASKS` list (it'll be
   pulled into `ALL_TASKS` automatically via `tasks.py`).
3. Run `python -m harness_bench verify-gold --task <new_id>` to make
   sure the verifier accepts the gold solution.
4. Run `python -m harness_bench run --task <new_id>` for an end-to-end
   sanity check against the live model.
