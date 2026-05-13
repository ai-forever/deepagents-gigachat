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
| `__main__.py` | CLI: `list`, `run`, `run-cli`, `verify-gold`. |

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

### Verifying without an LLM

Each task carries `gold_files` — the workspace state a "perfect" agent
would produce. The `verify-gold` command applies the gold solution and
runs the verifier without ever calling the model. Handy when adding new
tasks to make sure the verifier accepts a correct solution:

```bash
uv run python -m harness_bench verify-gold
```

## Results

### Final summary (200 tasks, profile v3 + `think`)

After the 150-task wave the bench was extended with 50 significantly
harder tasks (`tasks_extreme.py`, 151–200): composite CSV/SQLite/XLSX/JSONL
pipelines, archives (zip/gzip/tar), project-wide refactors, algorithms
with pytest (quicksort, LRU cache, linked list, tree inorder, priority
queue), statistics (rolling average, histogram, z-score, percentiles,
pivot table), XML/markdown with YAML front-matter, and hard composites —
three-way joins, hourly log aggregation, dead-function detection. All
200 tasks pass `verify-gold`.

| Date | Configuration | Concurrency | Result | % | Δ vs no-profile |
| --- | --- | --- | --- | --- | --- |
| 2026-05-13 | no profile | 5 | 134 / 200 | 67.0 % | — |
| 2026-05-13 | **profile v3 + `ThinkToolMiddleware`** (pinned) | 5 | **153 / 200** | **76.5 %** | **+19 (+9.5 pp)** |

#### Diverging tasks

**27 tasks passed only with the profile** (profile actually helped):

```
task_11_count_py             task_113_xlsx_update_cell
task_14_sum_numbers          task_117_json_to_yaml
task_21_rename_file          task_143_grep_largest_file
task_22_delete_file          task_144_grep_duplicate_funcs
task_25_sort_lines           task_147_log_filter_404
task_39_reverse_lines        task_149_sqlite_to_json
task_47_dedupe_lines         task_159_unzip_extract
task_48_append_eof_each      task_161_gzip_compress
task_56_move_to_subdir       task_170_impl_lru_cache
task_85_add_logging_import   task_179_csv_zscore_outliers
task_98_count_unique         task_181_csv_cumsum
task_102_csv_filter_adults   task_182_csv_group_agg
task_186_find_call_sites     task_188_csv_three_way_join
                             task_196_xlsx_to_csv_and_json
```

Recognisable clusters:
- **xlsx / archive / sqlite** — all three xlsx tasks (113/196), both
  archive tasks (159/161), sqlite export (149): the profile nudges the
  agent into `execute` + Python instead of trying to read the file by
  hand;
- **rename / move / two-step ops** (21, 22, 56, 188) and "process every
  line" (47, 48, 25, 39, 98) — the same kinds of wins we saw on the
  150-task bench;
- **harder algorithms and analytics** (170 LRU, 179 z-score, 181/182
  CSV aggregates) — only surfaced at this scale, and the profile wins
  here too.

**8 tasks passed only without the profile** (profile hurts):

```
task_24_add_header_comment      task_146_log_count_5xx
task_42_snake_case              task_178_csv_pivot_count
task_51_count_total_lines       task_190_concat_dedupe_sort
task_82_add_csv_column          task_191_sqlite_revenue_report
```

Dominant patterns:
- tasks where `think` pulls the agent into a long `read → think → edit →
  think → ...` loop that hits `GRAPH_RECURSION_LIMIT` or a model-node
  exception (`task_24`, `task_146`);
- aggregates where, with the profile, the model prefers "read and count
  in my head" and then slips on a single arithmetic step (`task_51`,
  `task_178`, `task_190`, `task_191`).

**39 tasks fail in both configurations** — the model's ceiling on the
current prompt: mostly `model`-node exceptions on tricky tool outputs
(XML parsing, multi-key JSON, markdown front-matter), some pytest tasks
with intricate structure (`task_171_impl_linked_list`,
`task_172_impl_tree_inorder`, `task_193_impl_memoize`),
`task_187_dead_functions`, and a few hard composites.

Net effect of the profile: **+27 wins − 8 regressions = +19**, which
matches the 67.0 % → 76.5 % bump. That is **larger** than the +7..+10
gap on 150 tasks, and **much larger** than the +5/+6 on 100. The trend
is consistent: as task diversity grows, the profile's advantage grows
with it.

### Intermediate summary (150 tasks, profile iterations)

After the 100-task wave the bench was extended with 50 harder tasks
(`tasks_hard.py`, 101–150): CSV / Excel / JSON / JSONL / YAML / INI /
TOML / SQLite, writing and running Python code (incl. pytest), wide
`grep` across 10+ files, Apache log parsing. All 150 tasks pass
`verify-gold`.

| Date | Configuration | Concurrency | Result | Δ vs no-profile |
| --- | --- | --- | --- | --- |
| 2026-05-13 | no profile (run 1) | 5 | 103 / 150 | — |
| 2026-05-13 | no profile (run 2, sanity check) | 5 | 106 / 150 | — |
| 2026-05-13 | **profile v3 + `ThinkToolMiddleware`** (pinned) | 5 | **113 / 150** | **+7 / +10** |

#### Diverging tasks

22 tasks passed **only with the profile** (it actually helped):

```
task_14_sum_numbers          task_119_yaml_bump_version
task_21_rename_file          task_120_ini_add_section
task_25_sort_lines           task_133_impl_factorial
task_33_find_max             task_140_grep_emails
task_34_filter_errors        task_147_log_filter_404
task_39_reverse_lines        task_149_sqlite_to_json
task_42_snake_case           task_108_csv_dedupe
task_47_dedupe_lines         task_111_xlsx_extract_b2
task_56_move_to_subdir       task_112_xlsx_sum_column
task_86_extract_numbers      task_113_xlsx_update_cell
task_114_jsonl_sum_amount    task_117_json_to_yaml
```

Recognisable clusters: **rename / move / two-step ops** (`task_21`,
`task_56`), **xlsx 3-of-3** (111/112/113), **YAML / INI / SQLite /
JSONL conversions**, and "process every line" tasks (`task_34`,
`task_42`, `task_140`). These are exactly the cases prompt v3 was
tuned for (sections `Two-step operations`, `Process EVERY line/file`,
clear formatting rules).

12 tasks passed **only without the profile** — regressions the profile
introduces:

```
task_20_move_function   task_101_csv_mean_score
task_24_add_header_comment   task_102_csv_filter_adults
task_26_add_json_key    task_106_csv_group_count
task_30_add_todo        task_138_grep_yaml_with_key
task_51_count_total_lines  task_144_grep_duplicate_funcs
task_65_sum_floats      task_148_csv_to_xlsx
```

Two themes dominate: (1) tasks where the `think` loop drives
`GRAPH_RECURSION_LIMIT` or a model-node exception (`task_26`,
`task_30`, `task_24`); (2) CSV tasks with many rows (`task_101`,
`task_102`, `task_106`) — with the profile the model more often picks
"read and count in my head" and slips on the arithmetic, while
without the profile it more often runs `python -c '...'` via
`execute`. That's a direction for further prompt work.

25 tasks **fail in both configurations** — the model's ceiling on the
current prompt: refactors like `task_20_move_function`, strict dict
formats (`task_55_add_conftest`), some pytest tasks, some log
tasks.

Net profile effect vs the first no-profile run: **+22 wins − 12
regressions = +10**, i.e. 113 / 150 vs 103 / 150. The second
no-profile run yielded 106 / 150 (±3 task flakiness between single
runs is expected without averaging), so the durable read on the
profile is **+7..+10**. The shape is the same as on the 100-task
bench (+5/+6); the gap widens with task diversity rather than washing
out — +5/+6 on 100 became **+7..+10** on 150.

### 100-task bench (history before the extension)

| Date | Configuration | Concurrency | Result | Δ vs no-profile |
| --- | --- | --- | --- | --- |
| 2026-05-13 | no profile (run 1) | 1 | 83 / 100 | — |
| 2026-05-13 | no profile (run 2, repeatability check) | 5 | 82 / 100 | — |
| 2026-05-13 | profile v3 + `ThinkToolMiddleware` | 1 | 88 / 100 | +5 / +6 |

The pinned configuration is **profile v3 with the `think` tool**:
- `deepagents_gigachat/prompts.py` — short v3 prompt with sections
  Two-step operations / Counting / Files / Search / Shell;
- `deepagents_gigachat/harness_profile.py` — reworked descriptions for
  `write_file` / `edit_file` / `grep` / `execute`, with
  `extra_middleware=(ThinkToolMiddleware(),)`.

### Intermediate summary (60 tasks, profile iterations)

| Date | Configuration | Result |
| --- | --- | --- |
| 2026-05-13 | no profile | 46 / 60 |
| 2026-05-13 | profile v1 (old version, `think` disabled) | 42 / 60 |
| 2026-05-13 | profile v2 (new prompt + new tool descriptions, `think` disabled) | 49 / 60 |
| 2026-05-13 | profile v3 (explicit two-step ops, "process everything", no re-read), `think` disabled | 51 / 60 |
| 2026-05-13 | **profile v3 + `ThinkToolMiddleware`** | **52 / 60** |

Details on the 60-task runs are in the "60 tasks: profile iterations"
section below. The 30-task historical summary (run on May 12) sits right
after it.

### Short historical summary (30 tasks)

| Date | Configuration | Result |
| --- | --- | --- |
| 2026-05-12 | `deepagents` + `langchain-gigachat` + GigaChat harness profile (prompt + `think` + tool overrides) | **22 / 30** |
| 2026-05-12 | same, but `ThinkToolMiddleware` disabled (prompt + tool overrides only) | **23 / 30** |
| 2026-05-12 | `deepagents` + `langchain-gigachat` without the harness profile | **26 / 30** |

All three runs used the same `GigaChat-3-Ultra` model via
`gigachat.ift.sberdevices.ru/v1`, the same `uv run python -m
harness_bench run` command with the default `--recursion-limit 80`, no
retries.

On this 30-task sample the `deepagents-gigachat` profile was a net loss:
without it the score was reliably higher. Disabling `think` inside the
profile only gained +1 (and removed one flaky task) — not enough to
offset the regression against bare `deepagents`. So the problem wasn't
`think`, it was the custom prompt and/or the
`write_file`/`edit_file`/`grep`/`execute` description overrides. Each
configuration was a single run with some flakiness — strict conclusions
need averaging across multiple runs — but the direction was
unambiguous. Per-run details follow.

### 2026-05-12 — with the `deepagents-gigachat` profile (`22 / 30`)

Environment: `deepagents 0.5.7`, `langchain-gigachat 0.5.1`,
`deepagents-gigachat 0.0.1a2` registered as the
`deepagents.harness_profiles` entry point and picked up automatically.
Total wall-time ≈ 5.5 minutes.

Failed tasks:

| Task | Reason |
| --- | --- |
| `task_11_count_py` | flake — failed in the main run with an exception, on retry `PASS` in 4.4 s |
| `task_13_count_csv_lines` | `GRAPH_RECURSION_LIMIT` — the agent got stuck in a tool loop |
| `task_20_move_function` | `a.py still contains forbidden: ['def helper', "'help'"]` — function copied to `b.py` but not removed from `a.py` |
| `task_21_rename_file` | `oldname.txt still exists` — content moved to `newname.txt`, original file left behind |
| `task_24_add_header_comment` | exception inside langgraph's `model` node |
| `task_25_sort_lines` | `sorted.txt lines differ` — line set or order didn't match expected |
| `task_26_add_json_key` | flake — failed in the main run, on retry `PASS` in 30.8 s |
| `task_30_add_todo` | `GRAPH_RECURSION_LIMIT` — another tool loop |

By type the eight failures break down as:

- **Real model misses** (the agent finished but did the wrong thing):
  `task_20`, `task_21`, `task_25` — all three about "delete the old
  thing after creating the new one" or about sorting.
- **`GRAPH_RECURSION_LIMIT` (80 steps)**: `task_13`, `task_30`.
- **Exception during a model call**: `task_24`.
- **Flakes** (passed on retry without code changes): `task_11`,
  `task_26`.

If you only count stable failures (drop the flakes), 24 / 30 are
reliably green.

### 2026-05-12 — profile active, `think` disabled (`23 / 30`)

Entry point and `register_harness()` are kept in place, but
`extra_middleware` in `deepagents_gigachat/harness_profile.py` is
switched from `(ThinkToolMiddleware(),)` to `()`. Verified that for a
fresh `GigaChat` instance `_harness_profile_for_model` returns a
profile with the custom `base_system_prompt` and the
`write_file`/`edit_file`/`grep`/`execute` overrides, with
`extra_middleware` empty.

Failed tasks:

| Task | Reason |
| --- | --- |
| `task_13_count_csv_lines` | `GRAPH_RECURSION_LIMIT` (114 s) |
| `task_20_move_function` | `a.py still contains forbidden: ['def helper', "'help'"]` — "half done" again |
| `task_21_rename_file` | `oldname.txt still exists` |
| `task_24_add_header_comment` | exception inside langgraph's `model` node |
| `task_25_sort_lines` | `sorted.txt lines differ` |
| `task_26_add_json_key` | `GRAPH_RECURSION_LIMIT` (54 s) |
| `task_30_add_todo` | `GRAPH_RECURSION_LIMIT` (93 s) |

Compared to "with the profile + `think`" (`22 / 30`):

- the flaky `task_11_count_py` dropped out (without `think` it passed
  in 27.8 s — slow, but it passed);
- `task_26_add_json_key` now reliably fails with `GRAPH_RECURSION_LIMIT`
  instead of being a flake/exception;
- the other six stable failures match one-to-one (`task_13`, `task_20`,
  `task_21`, `task_24`, `task_25`, `task_30`).

The takeaway: `think` was not the culprit. "Profile with `think`" and
"profile without `think`" produce nearly identical failure sets, and
both are visibly worse than "no profile at all". The regression came
from our custom prompt and/or the file-tool description overrides.

### 2026-05-12 — no profile (`26 / 30`)

For this run the `deepagents.harness_profiles` entry point in
`pyproject.toml` was disabled (commented out) and the package was
reinstalled so the `.dist-info` no longer pulled in the profile
automatically. The `register_harness()` call in `runner.py` was also
removed. Verified that for a fresh `GigaChat` instance
`_harness_profile_for_model(model, None)` returned an empty
`HarnessProfile()`: no custom `base_system_prompt`, no tool description
overrides, no `think` middleware. That is "plain" `deepagents` with the
default `BASE_AGENT_PROMPT` and stock file-tool descriptions. After the
run, the profile and entry point were restored — the current pinned
state of the repository matches the third run (profile active, `think`
disabled).

Failed tasks:

| Task | Reason |
| --- | --- |
| `task_14_sum_numbers` | exception inside langgraph's `model` node |
| `task_21_rename_file` | `oldname.txt still exists` — again didn't delete the source after copying |
| `task_25_sort_lines` | `sorted.txt lines differ` |
| `task_30_add_todo` | `GRAPH_RECURSION_LIMIT` (80 steps) |

### What changed without the profile

- **Stopped failing** (4 tasks): `task_11_count_py`,
  `task_13_count_csv_lines`, `task_20_move_function`,
  `task_24_add_header_comment`, `task_26_add_json_key` — five wins minus
  one new failure = net +4. In particular `task_20` ("move the function
  from a.py to b.py") with the profile reliably ends up "half done",
  while without it it goes through cleanly.
- **Started failing again or for the first time** (1 task):
  `task_14_sum_numbers` — failed inside langgraph's model node; passed
  with the profile.
- **Fail in both configurations**: `task_21_rename_file`,
  `task_25_sort_lines`, `task_30_add_todo`. Those are "real" model/task
  issues — the profile neither helps nor hurts.

That three-run 30-task series ended on the conclusion "the profile is
worse than its absence". After it:
- the bench was expanded from 30 to **60 tasks** (`tasks_extra.py`,
  tasks 31–60);
- a v1 → v2 → v3 profile iteration series was run; see below.

### 60 tasks: profile iterations

Goal of the series — get the profile to produce a measurable win over
the no-profile configuration. The model, the task set, and the run
command are identical across runs; the only thing that changes between
iterations is the content of `deepagents_gigachat/prompts.py` (system
prompt) and `deepagents_gigachat/harness_profile.py` (tool
descriptions).

#### 2026-05-13 — no-profile baseline (`46 / 60`)

The entry point in `pyproject.toml` is commented out and
`register_harness()` is removed from `runner.py`. Plain `deepagents`.

Failed tasks: `task_13_count_csv_lines`, `task_14_sum_numbers`,
`task_20_move_function`, `task_21_rename_file`, `task_24_add_header_comment`,
`task_25_sort_lines`, `task_32_count_words`, `task_33_find_max`,
`task_34_filter_errors`, `task_38_trim_trailing_ws`, `task_39_reverse_lines`,
`task_41_count_todos`, `task_55_add_conftest`, `task_56_move_to_subdir`.

Main causes: exceptions in langgraph's `model` node on branchy
tool-call payloads, `GRAPH_RECURSION_LIMIT`, and "two-step" issues
(renames / conversions where the source file was kept).

#### 2026-05-13 — profile v1 (`42 / 60`)

Old prompt and old tool descriptions, `think` disabled. The profile
lost 4 tasks vs no-profile. Diff analysis showed that the profile
*hurts* wherever it triggers extra steps: post-change `read_file` for
"verification", `read → edit → read → edit` cycles, repeated `grep`
on the same substring. Those cycles burn the recursion budget and
occasionally trip the model on format exceptions.

#### 2026-05-13 — profile v2 (`49 / 60`) — **+3 over baseline**

Prompt rewritten end-to-end:

- removed the "after changes, re-read files to verify" paragraph (the
  one that triggered cycles);
- added the rule "one file, one read, then one edit";
- moved two-step ops ("move / rename / convert") into a dedicated
  "do both halves" section;
- tool descriptions rewritten in a more direct style with one concrete
  example each.

#### 2026-05-13 — profile v3 (`51 / 60`) — **+5 over baseline**

On top of v2:

- a `Two-step operations` block with concrete hints ("after `mv` — both
  halves are done in one call", "after `replace X with Y` there must be
  ZERO occurrences of X");
- a "process EVERY line / file / item" rule targeting tasks like
  `task_34_filter_errors`, `task_38_trim_trailing_ws`,
  `task_48_append_eof_each`;
- a `Counting / arithmetic` section with the rule "compute once, write
  once, do not double-check the same number twice".

Remaining 9 v3 failures:

| Task | Reason |
| --- | --- |
| `task_13_count_csv_lines` | `model`-node exception |
| `task_20_move_function` | "copy without delete" in `a.py` again |
| `task_24_add_header_comment` | `model`-node exception |
| `task_30_add_todo` | `GRAPH_RECURSION_LIMIT` |
| `task_32_count_words` | `model`-node exception |
| `task_38_trim_trailing_ws` | misses trailing whitespace on several lines |
| `task_41_count_todos` | arithmetic mistake |
| `task_51_count_total_lines` | arithmetic mistake |
| `task_55_add_conftest` | writes single quotes instead of double quotes inside a dict literal |

Four of those (`task_13`, `task_24`, `task_32`, `task_30`) are
infrastructural: langgraph exceptions on serialised tool calls or
recursion-limit hits; hard to fix via the prompt. The other five are
model misses that could be nudged via another prompt iteration or a
higher `--recursion-limit`. No further iterations were run in this
commit — the goal "the profile should help" was reached, and we
preferred to lock in the steady gain instead of drifting the prompt
to chase flaky tasks.

### 100 tasks: final A/B

After the 60-task series the bench was extended to 100 tasks
(`tasks_more.py`, tasks 61–100). The new tasks cover `.env` edits,
nested JSON, dataclass scaffolds, CSV operations, light regex
extraction, INI/TOML/YAML stubs, splitting CSV into per-row files.
All 100 pass `verify-gold`.

The profile (v3 + `think`) was matched against the no-profile
configuration in a single run each:

| Date | Configuration | Result |
| --- | --- | --- |
| 2026-05-13 | no profile | 83 / 100 |
| 2026-05-13 | **profile v3 + `think`** | **88 / 100** |

#### Per-task diff

**Passed only with the profile** (the profile actually helps) — 8 tasks:
`task_14_sum_numbers`, `task_21_rename_file`, `task_25_sort_lines`,
`task_39_reverse_lines`, `task_56_move_to_subdir`, `task_75_squash_blank`,
`task_81_swap_lines`, `task_86_extract_numbers`. That's exactly the
two-step (rename / move), "process every line" and list-of-lines tasks
that prompt v3's `Two-step operations` and `process EVERY line`
sections were aimed at.

**Passed only without the profile** (the profile gets in the way) — 3
tasks: `task_13_count_csv_lines`, `task_26_add_json_key`,
`task_30_add_todo`. All three are cases where the profile + `think`
combo picked up a long chain of tool calls and hit
`GRAPH_RECURSION_LIMIT` (`task_26` and `task_30`) or a `model`-node
exception (`task_13`). That's the price of `think`.

**Fail in both configurations** — 9 tasks: `task_20_move_function`,
`task_32_count_words`, `task_41_count_todos`, `task_55_add_conftest`,
`task_65_sum_floats`, `task_71_count_assert`, `task_92_count_chars` plus
a few format-mismatch details. These show the model's ceiling on the
current prompt: refactoring with renames, exact sums/counts, strict
dict formatting.

Net profile gain = +8 wins − 3 regressions = **+5**. The profile is
reliably useful — and this is no longer a single 30-task slice, it's
100 tasks of various shapes.

### Repeated no-profile run with concurrency 5

After the first 100-task run the runner learned to parallelise tasks
through `ThreadPoolExecutor`: each task already lives in its own
`TemporaryDirectory`, there's no shared state, so threading
parallelism for I/O-bound LLM calls gives roughly a 3–4× speedup in
wall-time. The flag is `--concurrency N` (default 1).

To gauge run-to-run noise, the no-profile configuration (plugin
disabled — entry point commented out, `register_harness()` not called)
was repeated on the same 100 tasks, this time with concurrency 5:

| Date | Configuration | Concurrency | Result |
| --- | --- | --- | --- |
| 2026-05-13 | no profile (run 1) | 1 | 83 / 100 |
| 2026-05-13 | no profile (run 2) | 5 | 82 / 100 |
| 2026-05-13 | **profile v3 + `think`** | 1 | **88 / 100** |

One task difference between the two no-profile runs (flake); the
average no-profile is ≈ 82–83 / 100. The profile beats each of them by
**+5 / +6** tasks — i.e. the advantage is not run-to-run randomness.

The parallel no-profile run finished in about 3 minutes instead of
~15 minutes serially. Parallelism only affects wall-time — tasks are
isolated (their own workspace) and threading doesn't change outcomes.

### Current pinned state

The final profile v3 + `think` is the "production" configuration for the
repository:
- `pyproject.toml` declares the `deepagents.harness_profiles → gigachat`
  entry point;
- `runner.py:build_agent` calls `register_harness()` explicitly (so the
  dependency is visible in code);
- `deepagents_gigachat/prompts.py` — short v3 prompt with sections
  Two-step operations / Counting / Files / Search / Shell;
- `deepagents_gigachat/harness_profile.py` — reworked descriptions for
  `write_file` / `edit_file` / `grep` / `execute`, with
  `extra_middleware` set to `(ThinkToolMiddleware(),)`.

To compare against the no-profile configuration: comment out the
`[project.entry-points."deepagents.harness_profiles"]` block in
`pyproject.toml`, reinstall the package (`uv pip install -e .`), and
remove the `register_harness()` call in `runner.py:build_agent`. To run
the profile without `think`: change
`extra_middleware=(ThinkToolMiddleware(),)` back to
`extra_middleware=()`.

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
