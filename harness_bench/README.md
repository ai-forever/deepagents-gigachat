# harness_bench

A small in-tree benchmark of **231 file-operation tasks** used to drive
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
| `tasks_extreme.py` | Tasks 151–205: hardest wave (composite pipelines, archives, project-wide refactors, algorithms with pytest, statistics, XML/markdown, three-way joins). |
| `tasks_diagnostic.py` | Tasks 206–221: diagnostic wave (paid-revenue reconciliation, inventory anomalies, pricing API migration, latency reconstruction, tar+hash manifests, interval merge, config precedence, markdown link audit, data-quality reports, TODO/FIXME triage, category rollups, email extraction, runtime config, SQL leaderboards, import migrations, log-level summaries). |
| `tasks_memory.py` | Tasks 222–231: memory wave (read/write/forget/refuse in `MEMORY.md`; auxiliary deliverables like `LICENSE`, `requirements-dev.txt`, `bio.txt`, `profile.json`). Exercises the agent's memory-discipline rather than file I/O. |
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

## Task categories (231 in total)

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
  162–166, 192, 208, 220): removing a deprecated function, moving a
  function between files, headers, multi-file renames, extracting
  constants into a module, splitting a module by class, converting to
  `@dataclass`, project-wide import rewrites.
- **Filesystem operations** (incl. 21–23, 30, 48, 60, 159–161, 198,
  210): rename, delete, append, copyright headers, gzip, zip
  create/extract, tar extract, rename a directory, tar manifests with
  hashes.
- **JSON / config** (incl. 26–28, 49, 54–55, 61–63, 68–70, 91, 115–122,
  185, 197, 212, 217, 218): adding a key, bumping a dependency, CSV ↔
  JSON, CSV ↔ TSV, conftest fixtures, swapping CSV columns, nested JSON
  edits, YAML/INI/TOML edits, YAML front-matter parsing, merging config
  precedence, building runtime config, email extraction.
- **Python implementation + pytest** (incl. 125–134, 167–174,
  193–195, 200, 211): `fib`, `factorial`, `is_palindrome`,
  `count_vowels`, `quicksort`, `binary_search`, `is_balanced`,
  `LRUCache`, `LinkedList`, `TreeNode + inorder`, `is_anagram`,
  `two_sum`, `memoize`, `Timer` context manager, `MyRange` iterator,
  `PriorityQueue`, `merge_intervals`.
- **Multi-file `grep`/`glob` search** (incl. 135–144, 186–187, 213,
  215): counting `import`/`def`/`assert` across a project tree,
  listing files containing a marker, finding duplicates, dead-function
  detection, markdown link audit, TODO/FIXME triage.
- **Excel (xlsx)** (incl. 111–113, 148, 158, 196): cell extraction,
  column sums, cell updates, CSV ↔ xlsx, per-sheet split.
- **SQLite** (incl. 123–124, 149, 191, 199, 207, 219): counts, sums,
  JOIN + CSV export, JSON export, filtered queries, inventory
  anomalies, paid leaderboards.
- **Apache log parsing** (incl. 145–147, 189, 209, 221): top IP, 5xx
  count, status filter, hourly aggregation, request-latency
  reconstruction, INFO/WARN/ERROR summaries.
- **Composite pipelines** (incl. 151–158, 188–190, 206, 214, 216):
  CSV → JSON aggregates with filter+groupby+sort, SQLite JOIN → JSON,
  xlsx → markdown report, three-way joins, multi-CSV concat + dedupe,
  paid-revenue reconciliation, customer data-quality reports, category
  revenue rollups.

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
# all 231 tasks in sequence
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

At the end the runner prints `Passed: N/231` and a one-line summary for
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

All runs use `--concurrency 5` on the current **231-task** set. The
`deepagents` rows use this repo (`uv run python -m harness_bench run`
for GigaChat, `run-openrouter` for OpenRouter models). The `free-code`
rows use Claude Code CLI v2.1.119.

| Date | Runner | Model | Harness adapt | Result | % |
| --- | --- | --- | --- | --- | --- |
| 2026-05-21 | `free-code` 2.1.119 | **Claude Opus 4.7** | yes (built-in + AGENTS.md inject) | **231 / 231** | **100 %** |
| 2026-05-22 | `free-code` 2.1.119 | **Claude Haiku 4.5** | yes (built-in + AGENTS.md inject) | **222 / 231** | **96.1 %** |
| 2026-05-22 | `deepagents` | **GigaChat-3-Ultra** (PROM, deepagents 0.6.3 + langgraph 1.2.1) | **yes (v9 + memory wiring)** | **195 / 231** | **84.4 %** |
| 2026-05-22 | `deepagents` | GigaChat-3-Ultra (PROM, deepagents 0.6.2) | no (pkg uninstalled + entry-point disabled) | 154 / 231 | 66.7 % |
| 2026-05-23 | `ouroboros` | GigaChat-3-Ultra (PROM) | no | 147 / 231 | 63.6 % |
| 2026-05-22 | `deepagents` | MiniMax-M2 (via OpenRouter) | no | 209 / 231 | 90.5 % |
| 2026-05-22 | `deepagents` | DeepSeek V3.2-exp (via OpenRouter) | no | 208 / 231 | 90.0 % |
| 2026-05-22 | `deepagents` | GLM-4.6 (via OpenRouter) | no | 206 / 231 | 89.2 % |
| 2026-05-22 | `deepagents` | DeepSeek V4 Flash (284B-A13B MoE, via OpenRouter) | no | 186 / 231 | 80.5 % |
| 2026-05-22 | `deepagents` | OpenAI gpt-oss-120b (120B dense, via OpenRouter) | no | 165 / 231 | 71.4 % |
| 2026-05-22 | `deepagents` | Qwen3-235B-A22B-Instruct-2507 (235B-A22B MoE, via OpenRouter) | no | 162 / 231 | 70.1 % |
| 2026-05-22 | `deepagents` | GLM-4-32B (32B dense, via OpenRouter) | no | 76 / 231 | 32.9 % |

The GigaChat-3-Ultra rows quantify the contribution of the
`deepagents_gigachat` profile on the current bench: **+41 tasks (154 →
195)** comes from the v9 profile + memory wiring on top of stock
`deepagents`. The baseline run was done by uninstalling the package
(removing the `gigachat` entry-point from `deepagents.harness_profiles`)
and renaming the local source folder so `register_harness()` is
neither auto-discovered nor importable.

The 2026-05-21 Opus run is logged in detail because all 12 initial
failures turned out to be artifacts of the bench rather than real
model errors, and the fixes apply to other adapters too:

1. **Adapter mismatch — `tasks_memory.py` (8 fails fixed)**. The
   memory block (222-231) depends on the `AGENTS.md` → `MEMORY.md`
   convention used by `deepagents` / Codex CLI / Cursor. The
   free-code CLI ships its own host-side memory at
   `~/.claude/projects/...` and does NOT auto-load workspace
   `AGENTS.md`, so the agent never saw the memory instructions and
   silently no-op'd most memory tasks. `runner_cli.py` now detects
   a Claude-Code-style CLI and injects the workspace `AGENTS.md`
   via `--append-system-prompt`.
2. **Ambiguous prompts (3 fails fixed)**. `task_213` now states
   `domain == urlparse.netloc` (host+port, so `localhost:3000`
   rather than `localhost`); `task_215` shows an example confirming
   `<text>` keeps the `TODO:`/`FIXME:` marker; `task_206` now
   requires `paid_usd` formatted with exactly two decimal places.
3. **Verifier bug — `task_231` (1 fail fixed)**. The README-wording
   regex only accepted first-person "работаю с Anthropic", broadened
   to `работ\w*\s+с`. Secret-leak scan also skips dot-dirs (those are
   adapter debug captures, not agent artifacts).
4. **Contradictory prompt — `task_175` (1 fail fixed)**. Prompt said
   "median, min, max — целые числа", but gold uses `statistics.median`
   = 22.5. Prompt now spells out `median = (a+b)/2 for even-length,
   not rounded`.

The same fixes affected GigaChat too: it jumped from 184/231 (on
`deepagents` 0.6.2 + earlier bench prompts) to 195/231 once the
prompts were clarified and the stack was upgraded to deepagents 0.6.3
+ langgraph 1.2.1. Memory part still scores poorly for GigaChat
(2/10), suggesting that the `memory=["/AGENTS.md"]` wiring added to
`runner.py` does not reach GigaChat-3-Ultra's prompt in a form the
model acts on (Opus reads it just fine via `--append-system-prompt`
into Claude Code).

### Historical results

Older runs on the /200 and /221 task sets, plus the v3 → v9 profile
evolution notes, are archived in
[`LEGACY_RESULTS.md`](LEGACY_RESULTS.md). Those numbers are **not
comparable** to the /231 rows above and should not be used for
cross-model comparison.

The `yes (built-in)` rows pick up harness profiles that ship inside
`deepagents` itself (currently `anthropic:claude-opus-4-7`,
`anthropic:claude-sonnet-4-6`, `anthropic:claude-haiku-4-5`, and a
few `openai:gpt-5.x-codex` keys).

## Adding a task

1. In one of the task modules (`tasks.py`, `tasks_extra.py`,
   `tasks_more.py`, `tasks_hard.py`, `tasks_extreme.py`,
   `tasks_diagnostic.py` — pick the one that fits the wave / difficulty)
   describe a `Task(...)` — id, prompt, `setup_files`, `gold_files`,
   `verifier`.
2. Wire it into the corresponding module's `*_TASKS` list (it'll be
   pulled into `ALL_TASKS` automatically via `tasks.py`).
3. Run `python -m harness_bench verify-gold --task <new_id>` to make
   sure the verifier accepts the gold solution.
4. Run `python -m harness_bench run --task <new_id>` for an end-to-end
   sanity check against the live model.
