# Operating-curve sweep: does size-calibration dominate at matched recall?

**Question.** The first fix-track run compared `profile_cal` (calibrated to the k=1 FMR at t=0.5) to
plain `profile`/`anchor` at a single fixed point each and found calibration traded away too much
recall. This sweep asks the sharper question: at the SAME recall, does a size-calibrated threshold
reach higher precision than a single global threshold applied uniformly to every cluster size — i.e.
does the calibrated-threshold *curve* dominate the global-threshold *curve*, rather than just sitting
at a different, stricter operating point on the same curve?

## Setup

- **Global-threshold curves** (`profile_t<T>`, `anchor_t<T>`): the same fixed threshold T ∈
  {0.5, 0.8, 0.95, 0.99, 0.999} applied to every cluster size, for both the whole-profile and
  founding-member-only comparison targets.
- **Size-calibrated curve** (`profile_cal_tau<τ>`): for τ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}, every bin
  k ∈ {1,2,4,8,16} (including k=1, unlike the first fix-track run which pinned k=1 at t=0.5) is fit so
  the pooled hard-negative FMR at that bin is ≤ τ, on the same 186 held-out entities as before
  (`experiments/fix/sweep/fit_thresholds_tau.py` → `thresholds_tau.json`).
- **Soft audit** (`profile_audit_soft`): `profile` at the fixed 0.5 threshold, plus: at cluster sizes
  4, 8 and 16, evict a non-founding member only if BOTH its founder-alone P(YES) < 0.05 AND its
  leave-one-out profile score (profile minus itself, capped at 15 members) P(YES) < 0.5 — a
  double-sided, stricter eviction rule than the first run's single-sided `profile_audit` (P(YES)<0.5
  vs. the founder alone).
- All 16 variants ran to completion for all 6 models on the same 80 confusable blocks
  (`experiments/fix/sweep/e2e_sweep.py`, one model-pair per H100 job, ~2h total wall time). As a
  reproduction check, `profile_t0.5`/`anchor_t0.5` reproduce the first fix-track run's `profile`/`anchor`
  numbers exactly (differences at floating-point precision, <1e-16) for every model
  (`data/derived/fix/sweep/sweep_summary.csv` vs. `data/derived/fix/summary.csv`).
- **Matched-recall precision gain**: for each model, the three curves are built from point estimates
  (and, for CIs, from 2,000 paired block-bootstrap resamples) and linearly interpolated in
  recall-precision space; a gain is reported only where the target recall (0.2, 0.3, 0.4) falls inside
  the *observed* recall range of both curves being compared — outside that range the comparison is not
  defined, since the curve carries no information there and the two curves can extrapolate to the
  requested recall with completely different implicit thresholds.

## Results

### Does `profile_cal` dominate the global-threshold `profile` curve at matched recall? No.

Of the 9 (model, target-recall) cells where both the `profile_cal` and `profile_global` curves cover
the target recall, the paired-bootstrap 95% CI on the precision gain excludes 0 in **0 of 9 cases** — every
comparison is statistically indistinguishable from no gain (`data/derived/fix/sweep/matched_recall_gains.csv`,
column `sig_vs_profile`). Point estimates are small and mixed in sign: Qwen2.5-14B at R=0.4 shows the
largest positive point estimate (+0.057, CI [−0.006, 0.102], still crossing 0), Qwen2.5-32B at R=0.3 shows
the largest negative point estimate (−0.027, CI [−0.105, 0.039]). The other 9 (model, target-recall)
combinations have no defined comparison because at least one curve never reaches that recall within the
five thresholds/taus tested (full ranges in `sweep_summary.csv`; e.g. Mistral-24B's `profile_global` curve
tops out at recall 0.024, far below any of the three target recalls, so none of its three cells are
defined). So where the question *can* be asked, the answer is consistently negative: calibrating the
threshold by cluster size does not buy extra precision beyond what a well-chosen single global threshold
already gets at the same recall — the earlier run's apparent precision gain from calibration was from
moving to a different (lower-recall) point on essentially the same curve, not from a better curve.

### Does `profile_cal` dominate the `anchor` curve at matched recall? Only demonstrated for one model.

`anchor`'s recall range is narrow (0.0001–0.299 across the 6 models' 5 thresholds, touching 0 for Mistral-24B
at t=0.999), so only 2 of the 18
(model, target-recall) cells have both curves covering the same recall — both at R=0.2, for Qwen2.5-7B
and Qwen3-32B. Qwen2.5-7B shows a real gain: +0.067 precision at matched R=0.2, CI [0.001, 0.120],
excluding 0. Qwen3-32B's gain at the same recall is smaller and not significant (+0.012, CI
[−0.037, 0.058]). The other 16 cells are undefined because `anchor`'s maximum achievable recall across
the tested thresholds falls below 0.3 for every model and below 0.2 for Mistral-24B and OLMo-2-32B
specifically. So the evidence for `profile_cal` beating `anchor` at matched recall is a single
significant data point (Qwen2.5-7B), not a pattern across models.

### Per-model precision gains at R=0.2/0.3/0.4 (point estimates; "—" = not defined)

| model | gain vs profile @0.2 | @0.3 | @0.4 | gain vs anchor @0.2 | @0.3 | @0.4 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B   | −0.015 (ns) | — | — | **+0.067** (sig) | — | — |
| Qwen2.5-14B  | — | +0.005 (ns) | +0.057 (ns) | — | — | — |
| Qwen2.5-32B  | +0.005 (ns) | −0.027 (ns) | — | — | — | — |
| Qwen3-32B    | −0.008 (ns) | −0.008 (ns) | +0.017 (ns) | +0.012 (ns) | — | — |
| OLMo-2-32B   | +0.015 (ns) | — | — | — | — | — |
| Mistral-24B  | — | — | — | — | — | — |

("ns" = 95% paired-bootstrap CI includes 0; "sig" = CI excludes 0. Full point estimates, CIs and the
fraction of bootstrap resamples where the interpolation was defined are in `matched_recall_gains.csv`.)

### Does the soft audit improve `profile`'s F1? No — it is close to a no-op, and the one real effect is negative.

`profile_audit_soft` changes precision by less than 0.005 in magnitude relative to `profile_t0.5` for 5 of
6 models (largest |Δ|=0.0041, Qwen2.5-14B), with the 95% paired-bootstrap CI including 0 for all of them;
the sixth, Mistral-24B, shows ΔP=+0.0021 with the CI's lower bound landing exactly at 0.0000 (CI
[0.0000, 0.0067]) — a boundary case, not a clean exclusion of 0, and in any case on a base recall of only
0.024, so practically negligible either way. Recall drops for 5 of 6 models (Mistral-24B's recall rises
negligibly instead, ΔR=+0.0006, CI [0.0000, 0.0018], again boundary-touching), reaching clear significance
only for Qwen2.5-14B (−0.0141, CI [−0.0269, −0.0039]). Net F1 is flat-to-negative everywhere: the
only clearly significant ΔF1 is Qwen2.5-14B's −0.0109 (CI [−0.0218, −0.0027]); every other model's ΔF1 CI
includes 0, including Mistral-24B's ΔF1=+0.0011 whose CI again has its lower bound exactly at 0.0000
(CI [0.0000, 0.0035]) rather than a clean exclusion. The stricter double-sided eviction rule
(founder-alone P(YES)<0.05 AND leave-one-out P(YES)<0.5) triggers nearly as many audit checks as the
original single-sided `profile_audit` from the first fix-track run (23–277 events per model here vs.
28–440 there) but evicts far fewer records (2–18 per model here vs. 30–415 there —
`audit_soft_vs_profile.csv`, `sweep_summary.csv`), because requiring both conditions is a much narrower
trigger. The extra scoring cost is real, though: each audit
check now needs two LLM calls (founder-alone + leave-one-out) instead of one, pushing calls/record to
2.58–3.59, a per-model increase of +3% (Mistral-24B, which barely accretes) to +60% (Qwen3-32B) over plain
`profile`'s 2.24–2.50, for essentially no precision or F1 payoff.

## Practical recommendation

Neither size-calibration nor the softer audit rule changes the paper's conclusion from the first fix-track
run: plain whole-profile accretion's precision-recall trade-off is not reliably beaten by either fix once
compared at matched recall rather than at whatever operating point each method happens to land on. The
one genuine win found here — `profile_cal` beating `anchor` by +0.067 precision at matched R=0.2 for
Qwen2.5-7B — is a single model, not a general pattern, and the soft audit's only clearly significant effect
is a recall/F1 loss for Qwen2.5-14B; Mistral-24B's nominal precision/F1 gains are boundary-case bootstrap
results on a practically negligible base recall, not a real effect. If a precision-recall
improvement over plain accretion exists for these models, it was not found by threshold engineering alone
within the ranges swept here.

## How faithful / limitations

Each curve has only 5 points (5 global thresholds, 5 τ values), so the interpolated precision at a given
recall is a straight-line approximation between the two bracketing measured points, not a continuous fit;
narrower/denser sweeps could change the point estimates somewhat, though the consistent 0-of-9 non-significance
result against `profile_global` is unlikely to flip from denser sampling alone, since the paired-bootstrap CIs
already span the plausible range between the measured points. The target-recall comparisons are restricted to
where both curves are defined (see fractions above) — for most (model, target-recall) cells, especially
all three target recalls for Mistral-24B and all `anchor`-vs-`profile_cal` cells below R≳0.2, no comparison
could be made because `anchor` and (for Mistral) `profile_global` simply never reach that recall within the
thresholds tested; this is reported honestly as "undefined," not imputed. `profile_cal`'s calibration is
fit on the same 186 held-out entities as the first fix-track run (disjoint from the 80 evaluation blocks).
The soft-audit's founder_t=0.05/loo_t=0.5 thresholds were fixed per the task specification, not independently
tuned. All six models completed within the ~6h budget in about 2 hours of wall time (GPU queue was immediately
available, unlike the first fix-track run); no model or variant was dropped.
