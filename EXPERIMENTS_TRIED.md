# Hypotheses tried but not adopted

Track-record of profile experiments that *looked* sensible after analyzing
GigaChat-3-Ultra failures on `harness_bench`, but turned out to be neutral
or counter-productive when measured end-to-end. Each entry records the
hypothesis, the implementation, the bench delta, and what we learned —
so we don't burn time re-proposing the same idea later.

Methodology: each experiment is a single full run of 231 tasks against
GigaChat-3-Ultra (PROM endpoint, `deepagents` 0.6.3, `langgraph` 1.2.1)
at `concurrency=5`. Reference baseline: v9 + memory wiring +
`AgentsMdInjectMiddleware` (now v10) at **194/231 (84.0 %)** on
2026-05-23. Documented run-to-run noise on the bench is ±5 tasks, so any
delta within ±5 is statistically inconclusive even when directionally
positive — we treat ±5 as "neutral" and require a clearly positive trend
to keep a change.

---

## Exp 2 — Counting-strictness prompt update

**Hypothesis.** Seven failures looked like "model counted something
wrong and wrote the wrong number" (task_41 wrote 4 TODOs where there
were 5; task_135/139/146 grep-counter mismatches; task_75 squashed
blank lines off-by-one; task_143 grep-largest-file picked wrong file;
task_176 csv_rolling_avg off-by-one). Sounded like the model was doing
arithmetic in its head instead of through a `grep -c` / `wc -l` tool
call.

**Change.** Replaced the short "Counting / arithmetic" section in
`PYTHON_PROMPT` (`deepagents_gigachat/prompts.py`) with a much more
emphatic version that explicitly tells the model:

- NEVER count, sum, average, or compare in your head.
- Every number you write must come from the literal result of a tool
  call.
- The tool call for "count occurrences of X" is `grep -c X`; for "count
  lines" it's `wc -l < file`; for anything more complex, `write_file
  run.py` + `execute python run.py`.
- If you find yourself reasoning "I see N matches" without a tool, stop
  and use a tool.
- Compute once, write once.

**Result.** 192/231 (83.1 %), Δ **−2** vs the AGENTS.md-only baseline.
Inspecting the 8 counting-targeted tasks individually: 1 flipped from
FAIL to PASS (task_146_log_count_5xx); the other 7 stayed FAIL with the
exact same error pattern.

**What we learned.** The counting failures are **not** "model forgot to
use a tool". The model already uses `grep`/`wc`/`python` for counting;
the tool returns a correct number; then the model **writes a different
number into the output file** anyway. This is a data-attention /
transcription bug, not a tool-discipline bug. Reinforcing
tool-discipline prompts gives no leverage. The wasted ~700 tokens of
new prompt text also slightly displaced more useful context, which is
the likely explanation for the −2 swing.

**Status.** Reverted (`git checkout deepagents_gigachat/prompts.py`).
Don't propose more strict-counting prompt tweaks until/unless we find
a fix for the transcription gap (e.g. forcing the model to print the
number from the tool result *verbatim* into the output file via a
second tool call rather than reformulating it).

---

## Exp 3 — `ForbiddenLeftoverMiddleware` on edit_file

**Hypothesis.** Three failures (`task_20_move_function`,
`task_75_squash_blank`, `task_220_python_import_migration`) looked like
classic "two-step operations incomplete" — model edited the new
location but forgot to clean up the old one, so a forbidden pattern
(`def helper`, `utils.math`, etc.) remained in the file. Sounded like
something a tool-level self-check could catch: after a successful
`edit_file`, re-read the file, see if the `old_string` is still
present, and if so append a `[SELF-CHECK]` note nudging the model to
finish the cleanup.

**Change.** Added `ForbiddenLeftoverMiddleware` to
`deepagents_gigachat/harness_profile.py`. Wraps tool calls; when the
tool is `edit_file` and the call succeeded, reads the file from the
thread-local workspace, checks whether `old_string` is still present
AND `new_string` is absent (to avoid false positives from
`new_string` being a superset of `old_string`), and if both are true,
appends a `[SELF-CHECK]` paragraph to the `ToolMessage` content.
Provided both sync `wrap_tool_call` and async `awrap_tool_call`
implementations.

**Result.** 193/231 (83.5 %), Δ **−1** vs the AGENTS.md-only
baseline. The three targeted tasks stayed FAIL; one task that had been
PASSing (task_38_trim_trailing_ws) regressed to FAIL.

**What we learned.** The "two-step incomplete" failures aren't
"edit_file ran but didn't fully replace" — they're "model never called
edit_file on the old location at all". For `task_20_move_function` the
trace shows the model writes `helper` into `b.py` via `write_file` and
then declares the task done, never touching `a.py`. Middleware that
watches `edit_file` results has zero coverage on this failure mode.
The −1 swing was probably the added bytes in tool results displacing
context on an unrelated task (since smoke tests showed the middleware
*did* fire on at least one legitimate case).

If we want to address this category, the right shape is probably a
**post-completion verifier** that, when the model emits its final
"done" response, scans the workspace for patterns mentioned in the
user prompt (e.g. names of functions/files marked as the source side
of a "move X to Y") and challenges the model if those patterns are
still present. That is closer to bench-overfit territory though, so we
didn't pursue it now.

**Status.** Reverted. The middleware class itself was useful as an
exercise in `wrap_tool_call` + thread-local workspace plumbing — the
infrastructure is reused by `AgentsMdInjectMiddleware` in v10.

---

## Notes for future experiments

- Run-to-run noise on the 231-set is documented as ±5 tasks. Any single
  full run that shifts by less than that is **not** evidence either
  way. If a candidate change looks borderline (≤ ±5), repeat 3–5 times
  before deciding — Exps 2 and 3 above had a chance of being just bad
  rolls, but the lack of any visible per-task improvement (or any
  noticeable smoke-test win) made running multiples not worth it.
- Per-task heuristics, even when the failure pattern looks uniform,
  almost always have hidden displacement costs (token budget,
  conversation history bloat). The bar for a generic improvement is
  positive impact AND no observable regression on the categories it
  doesn't target.
- Bench-overfit guard: if an improvement would only fire on language /
  layouts / file names that exist in this bench, don't ship it.
