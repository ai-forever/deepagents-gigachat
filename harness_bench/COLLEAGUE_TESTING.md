# Colleague Testing Guide

Short handoff for testing the current model-router benchmark setup on branch
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

This verifies that the public routed runtime still works outside the benchmark:

```bash
uv run python examples/routed_workspace.py
```

Expected: the script prints the chosen route plus the resulting file content.

## 5. Quick smoke test for the model-router

This is the fastest useful check before a full run:

```bash
uv run python -u -m harness_bench run-router \
  --router-mode model \
  --no-routing-hints \
  --task task_08_bump_version \
  --task task_10_bump_pyproject \
  --task task_101_csv_mean_score \
  --task task_129_implement_passing \
  --concurrency 1
```

Why this set:

- `task_08_bump_version` checks a prompt-only code edit that should route to `deep`
- `task_10_bump_pyproject` checks a prompt-only config edit that should route to `deep`
- `task_101_csv_mean_score` checks a structured data task that should stay in `direct`
- `task_129_implement_passing` checks a pytest/code task that should route to `deep`

Expected: all 4 pass.

## 6. Main benchmark checks

### A. Primary run: prompt-only model-router

This is the most important command for the current work. It disables benchmark
task tags and routes from prompt text alone, with the model-router turned on:

```bash
uv run python -u -m harness_bench run-router \
  --router-mode model \
  --no-routing-hints \
  --concurrency 5
```

Recent result on my side: `199 / 200`.

Current single known failure on my side:

- `task_187_dead_functions`

### B. Comparison run: prompt-only rules-router

This is useful as a baseline against the deterministic router:

```bash
uv run python -u -m harness_bench run-router \
  --router-mode rules \
  --no-routing-hints \
  --concurrency 5
```

Recent result on my side: around `196-197 / 200`.

### C. Optional comparison: benchmark-adapter routing with internal hints

This is less important for the current work, but still useful as a reference:

```bash
uv run python -u -m harness_bench run-router --concurrency 5
```

Recent result on my side: around `195 / 200`.

## 7. What to report back

Please send:

- the exact command you ran
- the final `Passed: N/200`
- the list of failed task ids
- whether the run used `--router-mode model` or `--router-mode rules`
- whether the run used `--no-routing-hints`
- whether this was a clean branch or had local changes

Good comparison template:

```text
command: uv run python -u -m harness_bench run-router --router-mode model --no-routing-hints --concurrency 5
result: 199/200
failures: task_187_dead_functions
notes: clean branch / no local changes
```

## 8. Important note about variance

These runs are not fully deterministic. Exact failing tasks may move a bit
between runs, especially on the `direct` controller branch and on longer `deep`
pytest tasks. The main signal we care about is:

- whether the model-router stays close to `199/200`
- whether any failures look like genuine routing mistakes versus downstream
  execution errors
