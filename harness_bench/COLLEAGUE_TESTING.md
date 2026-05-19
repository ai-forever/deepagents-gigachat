# Colleague Testing Guide

Short handoff for testing the routed runtime and benchmark on the branch
`codex/gigachat-harness-router-profile`.

## 1. Get the branch

```bash
git fetch origin
git checkout codex/gigachat-harness-router-profile
git pull
```

## 2. Install dependencies

From the repository root:

```bash
uv sync
```

## 3. Configure GigaChat credentials

Use either:

```bash
export GIGACHAT_CREDENTIALS=...
```

or:

```bash
export GIGACHAT_USER=...
export GIGACHAT_PASSWORD=...
```

These can also live in `.env` in or above the repo root.

## 4. Optional sanity check

This verifies that the new public runtime works outside the benchmark:

```bash
uv run python examples/routed_workspace.py
```

Expected: the script prints the chosen route plus the resulting file content.

## 5. Quick smoke test

This is the fastest useful benchmark check:

```bash
uv run python -u -m harness_bench run-router \
  --no-routing-hints \
  --task task_01_create_hello \
  --task task_10_bump_pyproject \
  --task task_101_csv_mean_score \
  --task task_129_implement_passing \
  --concurrency 1
```

Why this set:

- `task_01_create_hello` checks a simple prompt-only `direct` route
- `task_10_bump_pyproject` checks a prompt-only `deep` route
- `task_101_csv_mean_score` checks a data task in `direct`
- `task_129_implement_passing` checks a pytest/code task in `deep`

Expected: all 4 pass.

## 6. Main benchmark checks

### A. Clean prompt-only routing

This is the most important run for the current work, because it disables
benchmark task tags and routes from prompt text alone:

```bash
uv run python -u -m harness_bench run-router --no-routing-hints --concurrency 5
```

Recent result on my side: around `196-197 / 200`.

### B. Benchmark-adapter routing with internal hints

This is useful as a comparison point:

```bash
uv run python -u -m harness_bench run-router --concurrency 5
```

Recent result on my side: around `195 / 200`.

## 7. What to report back

Please send:

- the exact command you ran
- the final `Passed: N/200`
- the list of failed task ids
- whether the run used `--no-routing-hints`

Good comparison template:

```text
command: uv run python -u -m harness_bench run-router --no-routing-hints --concurrency 5
result: 196/200
failures: task_20_move_function, task_143_grep_largest_file, ...
notes: no local code changes / ran on clean branch
```

## 8. Important note about variance

These runs are not fully deterministic. Exact failing tasks may move a bit
between runs, especially on the `direct` controller branch and on long `deep`
pytest tasks. The main signal we care about is the overall pass rate and
whether prompt-only routing stays close to the hinted benchmark route.
