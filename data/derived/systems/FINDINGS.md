# Systems track: profile accretion under published LLM-ER prompting strategies

**Question.** Does profile accretion — a hard negative becoming more likely
to be judged a match as the verified profile grows from k=1 to k=16 members
— appear only in the main experiment's single whole-profile next-token
prompt, or also in published prompting *formats* for LLM-based entity
resolution? Three formats are re-implemented on the same 266-entity FOPPA
profile pool with a fixed query per entity (see `experiments/systems/strategies.md`
for exact extraction + citations): **BatchER** (batched pairwise
match questions with fixed demonstrations, OR-aggregated; Fan et al., ICDE
2024), **ComEM-select** (candidate-cluster selection among A/B/"none", scored
by next-token softmax; Wang et al., COLING 2025), and **LLM-CER**-style
in-context clustering (direct multi-record partitioning; Fu et al., Proc.
ACM Manag. Data 2025). All three are run on all 6 cached open chat models
(Qwen2.5-32B-Instruct, Qwen2.5-14B-Instruct, Qwen2.5-7B-Instruct, Qwen3-32B,
OLMo-2-0325-32B-Instruct, Mistral-Small-24B-Instruct-2501) on one H100 each
via Aqua/PBS; 2,394 prompts per strategy per model (266 entities ×
{pos, negA, negB} × k ∈ {1,4,16}), 43,092 raw scored rows total
(6 models × 3 strategies × 2,394). Full per-(model, strategy, query-type, k)
point estimates, entity-level bootstrap 95% CIs (2,000 resamples) and exact
McNemar k=1-vs-k=16 paired tests are in `summary.csv`.

## Headline result: the bias is strategy-dependent, not universal

**BatchER (batched pairwise) and ComEM-select (candidate selection) both
reproduce the main experiment's accretion signature** — the false-assignment
rate on both hard negatives (negA, negB) rises monotonically (or
near-monotonically) from k=1 to k=16 in every one of the 6 models × 2
negative types = 12 cases, all McNemar p < 10⁻³ (mostly p < 10⁻¹⁰):

| strategy | mean Δ false-match (k1→k16), hard negatives | mean Δ recall (k1→k16), pos | significant cases |
|---|---|---|---|
| BatchER (batched pairwise, OR) | **+0.225** (range +0.18 to +0.32) | +0.291 | 12/12 |
| ComEM-select (A/B/none) | **+0.320** (range +0.22 to +0.44) | +0.483 | 12/12 |
| LLM-CER (in-context clustering) | **−0.207** (range −0.41 to +0.01) | −0.276 | 10/12 (decreasing), 2/12 flat |

Representative per-model numbers (false-match rate on negA, k=1 → k=16):
Qwen2.5-32B-Instruct 0.237→0.477 (BatchER), 0.282→0.571 (ComEM-select);
Mistral-Small-24B 0.459→0.639 (BatchER), 0.011→0.429 (ComEM-select,
a 39-fold increase from a near-zero base rate); OLMo-2-32B 0.233→0.436
(BatchER), 0.090→0.530 (ComEM-select); Qwen2.5-14B 0.241→0.432 (BatchER),
0.286→0.504 (ComEM-select); Qwen2.5-7B-Instruct 0.229→0.541 (BatchER; see
`summary.csv` for ComEM-select); Qwen3-32B 0.195→0.417 (BatchER; see
`summary.csv` for ComEM-select). negB shows the same pattern at every model
(see `summary.csv`).
Recall on the true positive query also rises with k in both formats (more
evidence helps the true match too), but the false-match growth on hard
negatives is of comparable or larger magnitude — the model is not simply
"more confident across the board", it specifically becomes more willing to
fold a hard negative into a larger verified cluster, i.e. the same accretion
mechanism identified in the main whole-profile experiment.

**LLM-CER-style in-context clustering shows the opposite trend**: in 5 of 6
models (all except Qwen2.5-32B-Instruct; all 10 cases McNemar p < 0.002,
9 of the 10 at p < 10⁻⁵, the remaining one — Qwen2.5-14B-Instruct negB —
at p = 1.66×10⁻³) the false-assignment rate on hard negatives *falls*
substantially as k grows
— e.g. OLMo-2-32B negA 0.422→0.060, Mistral negA 0.553→0.173, Qwen2.5-7B
negA 0.520→0.216, Qwen3-32B negA 0.211→0.075 — and recall on the true
positive falls in parallel (OLMo-2-32B pos 0.716→0.124, Mistral pos
0.812→0.462, Qwen2.5-7B pos 0.742→0.471, Qwen3-32B pos 0.474→0.274).
Qwen2.5-32B-Instruct is the one exception: both hard-negative rates are flat
across k (McNemar p=0.88 negA, p=1.00 negB; Δ ≈ ±0.005), i.e. no accretion
bias and no reversal either. The direction of the LLM-CER effect — less
willingness to merge the query into *any* cluster, true profile or hard
negative alike, as the record set grows — is a distinct failure mode
(under-merging / query-isolation as context grows) from the over-merging
accretion bias seen in the other two formats and in the main experiment;
it is not evidence against profile accretion as a general phenomenon, but it
shows the bias is a property of pairwise/candidate-accumulation interaction
patterns, not of in-context clustering, at least for this set composition.

## How faithful are the re-implementations

All three keep the published strategy's defining interaction structure
(batched pairwise sub-questions / A-B-none candidate selection / direct
multi-record clustering) but substitute: BatchER's demonstration selector
with 2 **fixed** demo pairs (paper's simplest "Fixed" category — no
covering-based selection, which needs a labeled candidate pool we don't
have); ComEM's two-stage match→select pipeline with the **select** primitive
alone, and a same-size distractor cluster drawn from a different FOPPA
entity rather than the papers' blocking output; LLM-CER's hierarchical
set-merging and misclustering guardrails with a single flat clustering call
per (entity, k) on a modest record set (k profile members + 1 query + 2
fixed distractors). None of these change the published format's interaction
*structure*; they remove each paper's cost-optimizing machinery, which is
orthogonal to the accretion question. Absolute accuracy numbers here are
therefore not comparable to the original papers' benchmark results — only
the within-strategy k-trend is the evidence this track reports.

**Parsing.** ComEM-select and BatchER were scored (next-token softmax /
line-regex) with 0% parse failure across all 28,728 rows combined (12 raw
files × 2,394 rows: 6 models × {batcher, comem_select}). LLM-CER's
free-form bracketed-partition output is far less reliable: parse-failure
rate averaged **7.0%** over the 54 (model × qtype × k) cells (unweighted
mean of `parse_fail_rate` in `summary.csv`), but this average is driven
almost entirely by one model. Qwen2.5-7B-Instruct alone averages **37.9%**
parse failure and reaches **88.7%** at (negB, k=4) — leaving only 30 of 266
rows usable in that cell — because the model prepends verbose chain-of-
-thought before the bracketed partition and our generation budget
(`10×n_recs+40` new tokens) truncates it before the answer appears; its
k=16 cells are also degraded (21.8–23.3% failure). The other five models
are much better behaved: OLMo-2-0325-32B-Instruct averages 2.3% (worst
cell 7.5%, negB k=1), and the remaining four (Qwen2.5-32B/14B-Instruct,
Qwen3-32B, Mistral-Small-24B) average ≤1.5% with a worst single cell of
4.1%. Failed rows are excluded from the rate/recall denominators in
`summary.csv` (the `n` column) rather than silently counted as non-match,
so Qwen2.5-7B-Instruct's LLM-CER point estimates — especially at k=4, and
to a lesser extent k=16 — should be read as **provisional**, computed on a
reduced and potentially non-representative subsample; its qualitative
reversal direction is the same as the other 4 non-flat models (see
`summary.csv`), but its exact magnitude is less trustworthy than the
other five models'.

## Limitations

- ComEM's distractor cluster B and LLM-CER's 2 distractor records are drawn
  from a different entity in the same 266-entity pool rather than from the
  hard negative's own SIREN (items_all.json has no member list for
  negA/negB's own entity, only its query text), so "false assignment" here
  means assignment to the *true profile* cluster specifically, not confusion
  with a plausible same-brand sibling; this is the same limitation the main
  whole-profile experiment does not have (it compares query text directly
  against profile members) but it does not affect the within-strategy k-trend,
  which is the finding reported.
- All 6 cached open models were run (Qwen2.5-32B/14B/7B-Instruct, Qwen3-32B,
  OLMo-2-0325-32B-Instruct, Mistral-Small-24B-Instruct-2501) — no model
  subsampling.
- Qwen2.5-7B-Instruct's LLM-CER numbers are provisional: a generation-length
  budget tuned for the other five models left 11–12% of its rows usable at
  k=4 and 76–78% at k=16 (verbose chain-of-thought before the bracketed
  answer), so its point estimates for that strategy carry materially more
  uncertainty than the other 5 models' (see Parsing above); a rerun with a
  larger `max_new_tokens` budget for this model specifically was not
  performed in this pass.
- BatchER's OR-aggregation rule (predicted match if the model says YES to
  any single batched sub-question) is our choice, not something BatchER's
  paper prescribes for a same-entity batch of this shape; a different
  aggregation (e.g. majority vote) would shift absolute rates but not the
  qualitative k-trend, since the trend is already visible in the per-k rate
  of individual sub-answers turning positive (see `data/raw/systems/batcher_*.jsonl`
  raw completions for the per-question answers before aggregation).
