# Fix track: calibrated thresholds and a size-triggered audit for online linking

**Question.** Does a size-dependent decision threshold and/or a periodic re-score-against-founder
audit raise end-to-end online-linking precision at an acceptable recall cost, relative to the plain
whole-profile accretion rule (`profile`) and to the founding-member-only rule (`anchor`)?

## Setup

- **Threshold fitting** (`experiments/fix/fit_thresholds.py` → `experiments/fix/thresholds.json`):
  for each of the 6 models, the pooled hard-negative (negA+negB) false-match rate at the single-member
  baseline (`t=0.5`) on the 186 entities *not* used in `blocks_e2e.json` is the target; for nested
  profile sizes k∈{2,4,8,16} the smallest threshold that keeps the k-size FMR at or below that target
  is selected (capped at the largest value below the maximum score if the target is unreachable — this
  happened only for Qwen2.5-32B at k=8 and k=16). The fitted thresholds saturate close to 1.0 for most
  model/size combinations (e.g. Qwen2.5-14B: t=1.0 at every k≥2; OLMo-2-32B is the mildest, t=0.59–0.96).
  This already signals that enforcing the k=1 false-match rate at larger profile sizes is a very strict
  constraint for these models — their accretion-inflated scores on hard negatives leave little room
  below the ceiling.
- **Representations re-run with `experiments/fix/e2e_fix.py`** on the same 80 confusable blocks
  (2,572 records): `anchor`, `profile` (both at the fixed 0.5 threshold, for comparability), `profile_cal`
  (whole profile, threshold = the bin above keyed to the current cluster size), `profile_audit` (whole
  profile at 0.5, plus: whenever a cluster's size reaches 4, 8 or 16, every non-founding member is
  re-scored against the founding member alone and evicted into its own singleton cluster if P(YES)<0.5),
  `profile_cal_audit` (both). All five representations ran to completion for all 6 models (Qwen2.5-7B/14B/32B,
  Qwen3-32B, OLMo-2-32B, Mistral-24B) in one Aqua H100 job each (3 jobs, 2 models/job).
- **Metrics**: pairwise precision/recall/F1 from `pair_counts` (same definition as `scripts/build_assets.py`),
  share of output clusters containing more than one true SIREN ("mixed/contaminated clusters"), and LLM
  calls per input record (join-decision calls + audit re-scoring calls). 95% CIs are block-level
  bootstrap (2,000 resamples of the 80 blocks); the profile-vs-fix deltas use the *same* resampled block
  indices for both representations (paired bootstrap), so the CI reflects the within-block correlation.

## Results

### Calibration (`profile_cal`, `profile_cal_audit`) does not pay off

Relative to plain `profile`, calibrated thresholds raise precision by −0.02 to +0.18, with the 95% CI
excluding 0 for only 3/6 models (Qwen2.5-7B, Qwen2.5-14B, Qwen3-32B; the other three are not distinguishable
from no change). Recall drops by 0.015–0.26 for every model and the CI excludes 0 for all 6. Pairwise F1 is
lower than `profile` for every model with the CI excluding 0 in all 6 cases (mean ΔF1 = −0.15, median −0.16,
worst −0.27 for Qwen2.5-32B). `profile_cal_audit` is essentially the same story relative to `profile`
(precision CI excludes 0 for only 2/6 models; recall and F1 both exclude 0, i.e. are reliably worse, for
all 6; mean ΔF1 = −0.17): once the calibrated threshold is active, clusters rarely reach the size-4/8/16
audit trigger at all, so the audit adds little beyond the calibration itself. Against `anchor`, calibration
is inconsistent rather than a clean wash: ΔF1's 95% CI excludes 0 for 5/6 models — negative for Qwen2.5-7B,
Qwen2.5-32B, Qwen3-32B and Mistral-24B, positive for Qwen2.5-14B — and includes 0 only for OLMo-2-32B
(mean ΔF1 = −0.02, median −0.03). In short, forcing every profile size's false-match rate back down to the k=1 level is a threshold
so strict (saturating at ≈1.0 for 4 of 6 models beyond k=2) that it suppresses almost all genuine growth,
not just the contamination, and the net effect relative to the simplest baseline is model-dependent rather
than reliably neutral.

### The size-triggered audit (`profile_audit`) is a partial, not a net, fix

Relative to `profile`, the audit raises precision only marginally (−0.03 to +0.08, CI excludes 0 for only
1/6 models, Qwen2.5-14B) while it still gives back recall for most models (−0.001 to −0.13; CI excludes 0
for 5/6 models — Mistral-24B's CI, [−0.0041, 0.0010], includes 0), so pairwise F1 is below `profile` for
every model, with the CI excluding 0 for 5/6 (again all but Mistral-24B; mean ΔF1 = −0.06, median −0.06);
the audit never recovers the full profile-vs-anchor recall gain. But relative to `anchor`, the audit is the
only one of the three fixes that is consistently positive: F1 improves for all 6 models and the 95% CI
excludes 0 for every model including the smallest margin, Qwen2.5-7B (lo=+0.003, hi=+0.061; mean ΔF1 = +0.07,
range +0.01 to +0.16). The mechanism is visible in the per-model breakdown: for the three largest, most
accretion-prone models (Qwen2.5-14B/32B, OLMo-2-32B) recall rises significantly over `anchor` (CI excludes 0,
+0.05 to +0.15) while precision moves by at most ±0.05 with the CI crossing 0 for all but OLMo-2-32B
(whose CI is [−0.099, −0.005], a reliable but small drop), i.e. the audit lets the cluster keep most of `profile`'s recall advantage
over the founding-member-only rule while catching and evicting some of the false joins along the way. So
the audit sits strictly between `anchor` and `profile` on the precision-recall curve for every model: it
recovers part, not all, of plain accretion's recall benefit and gives back only a little precision, landing
above `anchor` but below `profile` in F1.

### Contaminated-cluster share moves little

The share of output clusters containing more than one true SIREN changes only slightly for most models
(≤1 percentage point across representations for Mistral-24B, OLMo-2-32B, and Qwen2.5-7B/32B; see
`summary.csv` mixed_share columns), but two models move more: Qwen2.5-14B spans 3.4 percentage points
(4.99%–8.43%, lowest under `profile_cal`, highest under `profile_audit`) and Qwen3-32B spans 2.6 points
(9.10%–11.72%, lowest under `profile_cal_audit`, highest under `anchor`). None of the three fixes
meaningfully or uniformly reduces contamination as measured at the cluster level; the F1 effects above are
driven mainly by the overall precision/recall trade, not by a systematic reduction in mixed clusters.

### Cost

Average LLM calls per input record: `anchor` 2.43, `profile` 2.37, `profile_cal` 2.45, `profile_audit` 3.00
(+27% over `profile`), `profile_cal_audit` 2.73 (+15%). The audit's extra cost is concentrated in the
largest, most accretion-prone models — under `profile_audit`, Qwen2.5-14B triggers 440 audit events with
415 evictions and Qwen2.5-32B triggers 345 events with 313 evictions, vs only 28 events / 30 evictions for
Mistral-24B, which barely accretes in the first place (full per-model/rep counts are in `summary.csv`'s
n_audit_events/n_evictions columns).

## Practical recommendation

Neither fix is a net improvement over plain whole-profile accretion by pairwise F1 on this benchmark.
The size-triggered audit is the more defensible of the two: it is the only fix that reliably beats the
conservative `anchor` baseline (positive ΔF1 for all 6 models) at a modest (~27%) extra-call cost, trading
part of accretion's recall gain for a comparable amount of precision. The calibrated-threshold fix, in the
form tested here (holding every larger profile size's hard-negative false-match rate to the single-member
rate), is too strict for these models — it collapses recall well before it stabilizes precision, and should
not be used as specified; a looser target (e.g. matching the k=1 true-positive *recall* rather than FMR, or
interpolating rather than hard-capping at k=16) would be a natural next calibration to test.

## How faithful / limitations

Thresholds were fit on the 186 FOPPA entities held out of the 80 online-linking blocks, so the calibration
and the evaluation never touch the same entities; Qwen2.5-32B's k=8/k=16 thresholds could not reach the
k=1 false-match-rate target even at the maximum observed negative score and are reported as capped
(`thresholds.json`, `capped: true`), so that model's `profile_cal`/`profile_cal_audit` numbers at large
cluster sizes are the most conservative threshold attainable from this fitting set, not a target-achieving
one. All five representations share the same 80 confusable blocks and stream order as the paper's main
online-linking run, so precision/recall are directly comparable to the `profile`/`anchor` numbers already
in `data/derived/e2e.csv`. As a reproduction check, the `anchor`/`profile` numbers recomputed here with
`e2e_fix.py` match the main run (`data/derived/e2e.csv`) to 3 decimals for 5 of 6 models; OLMo-2-32B differs
by up to 0.025 (e.g. profile recall 0.236 here vs 0.235 there, precision 0.743 vs 0.768), consistent with the
smaller batch size used here (`--bs 16` vs the main run's default 24) changing bf16 padding and hence the
exact logits for that model — the join decisions are identical algorithmically, not re-derived from a
different candidate-generation or scoring pipeline. The audit threshold for eviction is fixed at 0.5 (not
calibrated) in both `profile_audit` and `profile_cal_audit`; this was the specification given, not an
independently tuned choice.
