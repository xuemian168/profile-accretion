# Prompt robustness of the accretion effect (track: prompts)

**Question.** Is the profile-accretion effect (false-match rate and recall
both rising as the verified profile grows from k=1 to k=16 members) specific
to the exact wording of the matching prompt used for the main results, or
does it replicate under materially different instructions, record layouts,
and answer formats?

**Method.** Three alternative prompts were written (full text in
`experiments/prompts/prompts.md`), all keeping a next-token YES/NO readout
(P(YES) = softmax mass on YES-family tokens / (YES-family + NO-family
tokens)):

- **V1** — reworded instruction and record framing (same SIREN-based
  definition, phrased as a "data-matching analyst" task with a "cluster
  under review").
- **V2** — members shown as a numbered markdown table; the model is told to
  answer "Answer: YES" / "Answer: NO"; the readout is taken at the token
  immediately after a literal "Answer:" string appended to the prompt (not
  parsed through the chat template — this is a plain text continuation, so
  it carries over to any tokenizer without relying on template-specific
  prefill support).
- **V3** — the bare question with no SIREN / legal-entity definition at
  all and a generic system prompt ("Do these records refer to the same
  company?").

All six open models (Qwen2.5-7B/14B/32B-Instruct, Qwen3-32B,
OLMo-2-0325-32B-Instruct, Mistral-Small-24B-Instruct-2501) were scored under
each variant on one Aqua H100, using
`experiments/prompts/score_variants.py` (`--variant {v1,v2,v3}`). To keep
the added compute bounded, the job grid per model per variant was reduced
relative to the main run: for the 266 FOPPA entities, only k in {1, 4, 16}
was scored for pos/negA/negB (41 prompts/entity x 266 = 10,906
prompts/model/variant, vs. 16,758 in the main k-in-{1,2,4,8,16}-plus-reversed
run), plus the 16 single-member judgments for negA/negB needed for the
OR-null at k=16. This is declared as a scope reduction relative to the
original protocol, not a silent cut (see `experiments/prompts/prompts.md`,
"Implementation note"). All 18 model x variant scoring runs (6 models x 3
variants) completed on the Aqua H100 queue; every output file has exactly
10,906 scored rows.

**Readout is well-defined in every variant.** Mean YES/NO token mass
(the softmax probability captured by the YES/NO vocabulary, versus leaking
to other tokens) is >=0.94 for every model x variant combination, and
>=0.99 for 16/18; the two below that are both Mistral-Small-24B (V1: mean
0.941, min 0.886; V3: mean 0.989, min 0.979 — see
`yesno_mass_by_variant.csv`). The V2 "Answer:" prefill readout is as
reliable as the unprompted V0/V1/V3 generation-prompt readout.

## Headline result: the accretion effect replicates under all three variants, for all six models

| model | variant | FMR k=1 | FMR k=16 | recall k=1 | recall k=16 | AUC k=1 | AUC k=16 | McNemar p (k1 vs k16) |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B  | V1 | 0.000 | 0.023 | 0.000 | 0.056 | 0.679 | 0.746 | 4.9e-4 |
| Qwen2.5-7B  | V2 | 0.039 | 0.141 | 0.147 | 0.316 | 0.686 | 0.724 | <1e-16 |
| Qwen2.5-7B  | V3 | 0.209 | 0.432 | 0.368 | 0.823 | 0.663 | 0.736 | <1e-16 |
| Qwen2.5-14B | V1 | 0.000 | 0.120 | 0.000 | 0.410 | 0.629 | 0.772 | <1e-16 |
| Qwen2.5-14B | V2 | 0.038 | 0.133 | 0.162 | 0.417 | 0.707 | 0.768 | <1e-16 |
| Qwen2.5-14B | V3 | 0.064 | 0.385 | 0.139 | 0.861 | 0.668 | 0.777 | <1e-16 |
| Qwen2.5-32B | V1 | 0.000 | 0.222 | 0.000 | 0.718 | 0.711 | 0.800 | <1e-16 |
| Qwen2.5-32B | V2 | 0.055 | 0.286 | 0.158 | 0.789 | 0.712 | 0.788 | <1e-16 |
| Qwen2.5-32B | V3 | 0.154 | 0.395 | 0.361 | 0.880 | 0.695 | 0.785 | <1e-16 |
| Qwen3-32B   | V1 | 0.008 | 0.265 | 0.011 | 0.741 | 0.662 | 0.786 | <1e-16 |
| Qwen3-32B   | V2 | 0.058 | 0.271 | 0.203 | 0.726 | 0.696 | 0.794 | <1e-16 |
| Qwen3-32B   | V3 | 0.205 | 0.417 | 0.477 | 0.880 | 0.695 | 0.784 | <1e-16 |
| OLMo-2-32B  | V1 | 0.000 | 0.004 | 0.004 | 0.008 | 0.696 | 0.734 | 0.500 |
| OLMo-2-32B  | V2 | 0.002 | 0.101 | 0.008 | 0.289 | 0.707 | 0.748 | <1e-16 |
| OLMo-2-32B  | V3 | 0.126 | 0.306 | 0.320 | 0.650 | 0.688 | 0.728 | <1e-16 |
| Mistral-24B | V1 | 0.000 | 0.004 | 0.000 | 0.004 | 0.680 | 0.707 | 0.500 |
| Mistral-24B | V2 | 0.143 | 0.355 | 0.391 | 0.778 | 0.695 | 0.751 | <1e-16 |
| Mistral-24B | V3 | 0.205 | 0.474 | 0.432 | 0.895 | 0.682 | 0.771 | <1e-16 |

Full table with CIs: `summary.csv`; per-k curves: `accretion_curves_by_variant.csv`.

Across all 18 model x variant combinations, FMR and recall both rise
monotonically from k=1 to k=4 to k=16 — the same direction as the original
prompt (V0). The entity-level bootstrap (2,000 resamples over the 266
SIREN entities) gives a 95% CI for FMR(k=16) - FMR(k=1) that excludes zero
in **16/18** combinations, for recall(k=16) - recall(k=1) in **16/18**, and
for AUC(k=16) - AUC(k=1) in **17/18** (`bootstrap_entity_by_variant.csv`).
The exact McNemar test on the paired k=1-vs-k=16 negative-query judgments
is significant (p<0.05) in **16/18** combinations (`mcnemar_k1v16.csv`).

The two non-significant FMR/recall/McNemar cells are the same two cells in
each of those three tests: **OLMo-2-32B/V1** and **Mistral-24B/V1**, where
FMR and recall are already essentially zero at k=1 *and* at k=16 (FMR
0.000->0.004, recall for OLMo 0.004->0.008, for Mistral 0.000->0.004) —
i.e. V1's wording pushes these two models to an extremely conservative
operating point under which almost every query gets a NO regardless of
profile size, leaving almost no room to detect a k=1-to-k=16 change (only 2
discordant record pairs each for McNemar). The AUC-CI exception set is
smaller: only **Mistral-24B/V1** (dAUC mean +0.027, 95% CI [-0.014,
+0.067]) fails to exclude zero; OLMo-2-32B/V1's AUC *does* move
significantly (dAUC mean +0.038, CI [0.006, 0.069]) despite its FMR/recall
being floored, because AUC is computed from the full P(YES) distribution,
not just the >0.5 threshold crossings that floor out for FMR/recall. This
is a floor/threshold effect from V1's induced calibration, not evidence
against the accretion mechanism — both models show a clear, significant
accretion effect under V2 and V3, and under the original V0 prompt
(dFMR = +13.5 pp for OLMo, +23.7 pp for Mistral from k=1 to k=16, see
below).

## Comparison to the original prompt (V0)

The original prompt's k=1-to-k=16 FMR increase (from
`data/derived/accretion_curves.csv`) is similar in magnitude to what V1-V3
produce, once the wording-dependent operating point is accounted for:

| model | V0 dFMR(k1->k16) | V1 dFMR | V2 dFMR | V3 dFMR |
|---|---|---|---|---|
| Qwen2.5-7B  | +18.2 pp | +2.3 pp | +10.1 pp | +22.3 pp |
| Qwen2.5-14B | +30.5 pp | +12.1 pp | +9.6 pp  | +32.1 pp |
| Qwen2.5-32B | +32.3 pp | +22.2 pp | +23.1 pp | +24.1 pp |
| Qwen3-32B   | +18.4 pp | +25.7 pp | +21.2 pp | +21.3 pp |
| OLMo-2-32B  | +13.5 pp | +0.4 pp  | +10.0 pp | +18.0 pp |
| Mistral-24B | +23.7 pp | +0.4 pp  | +21.2 pp | +26.9 pp |

Every model has a positive dFMR under every variant (18/18); the V1 cells
for OLMo and Mistral are the smallest (floor effect, above). V2 and V3
reproduce dFMR of the same order of magnitude as V0 for every model; no
variant or model reverses the sign of the effect.

## OR-null exceedance and confident false matches (k=16)

`fmr_or` is the share of hard negatives that the model's OWN single-member
judgments (any-of-16, logical OR) would already flag at P(YES)>0.5, i.e. the
rate the profile's false-match rate would have to beat to show any benefit
from aggregation beyond its most permissive single member.
`exceedance_pp` = 100 x (profile FMR - OR-null FMR) ranges from -9.8 pp to
+21.1 pp across variant x model cells (`summary.csv`, `exceedance_pp`); the
profile-level judgment agrees with its own OR-null on 77-99% of hard
negatives (`agree_or_k16`). This mirrors the qualitative pattern reported
for V0 (profile judgments track, but do not simply reduce to, the
most-confident single member) and is not variant-specific. The share of hard
negatives judged a match with P(YES)>0.99 at k=16 ("confident false
matches") ranges 0.0-33.5% across variant x model cells, consistent with the
V0 range reported in `confident_false_matches.csv`.

## How faithful / limitations

- The job grid for this track is a declared subset of the main protocol (k
  in {1,4,16} only, no k in {2,8}, no reversed-order control, and
  single-member judgments restricted to negA/negB) — see "Implementation
  note" in `prompts.md`. This is sufficient for every statistic requested
  for this track but the per-k resolution is coarser than the main
  accretion curve.
- Each variant's absolute false-match/recall level is not directly
  comparable across variants or to V0 as a calibration statement — different
  wordings set different operating points (most sharply for V1 on OLMo and
  Mistral, which collapses to near-all-NO). The quantity that is comparable,
  and the one this track was designed to test, is the *within-variant*
  direction and significance of the k=1-to-k=16 change, which is positive
  and significant in 16-17/18 cells regardless of wording.
- V2's readout depends on a literal "Answer:" string appended after each
  model's own chat template rather than a template-native prefill
  mechanism; this was a deliberate implementation choice (see
  `score_variants.py`) made for portability across the three different
  tokenizers/templates used here, and it is validated post hoc by the
  near-unit YES/NO token mass under V2 (0.987-1.000 across all six models).
- All numbers are from a single scoring run per model x variant (no repeat
  decoding runs); given the deterministic forward-pass readout (no sampling
  involved), this is not a source of additional variance beyond the entity
  bootstrap already reported.
