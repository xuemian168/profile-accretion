> **Superseded (2026-10-27).** This file documents the first, quota-limited Sonnet run. The final closed-model results combine that run with a completion shard (Sonnet) and two Haiku 5.5 shards with thinking disabled (`haiku_shardA.jsonl`, `haiku_shardB2.jsonl`; `haiku_shardB.jsonl`, run with thinking on, is excluded). See `experiments/closed/summarize_closed.py`, `summary_full.csv`, `snowball_full.csv` and `coverage_final.csv`.

# Closed-model (Claude) track — FINDINGS

## Scope actually completed vs. planned

**Planned:** sample 100 entities x {k=1,4,16} x {pos,negA,negB} (900 profile prompts) plus
212 k=4 clean/contaminated snowball prompts, 3x each, for **two** Claude models
(claude-sonnet-5, then claude-haiku-5-5), about 6,672 requests in total through the Anthropic API.

**What ran in this first pass:** the sampling environment capped total usage at **2.0M tokens,
shared across models**. With a ~4.9k-token cached system prompt per call (independent of the short
YES/NO answer), the cap allows only ~350-400 calls' worth of budget before further calls fail with
a budget error, which is recorded in the `err` field of the raw outputs.

Run order: **claude-sonnet-5** was
sampled first, in shuffled order across all 1112 prompts, 3 samples/prompt
(queue of 3336 requests). The ceiling was hit at request **2610** of
3336 (our own usage-field sum at that point was ~14.0M because cache-read
tokens are also metered in the `usage` field but evidently weighted differently by the actual
ceiling check). A follow-up probe confirmed the cap applies across models: a single
**claude-haiku-5-5** call immediately returned the same budget error with **zero** tokens spent
on Haiku. **Haiku was therefore not run in this first pass; every Haiku number below is absent,
not zero** (the final Haiku results come from the later shards named at the top).

Sonnet coverage actually obtained (shuffled sampling order means the stopped run is a random
~78% subsample of the planned prompt set, not a systematic subset):
- Profile prompts (k x q): **683 / 900** prompts got ≥1 valid sample (75.9%); all 100 entities have ≥1 covered profile prompt.
- Snowball (clean/contaminated, k=4): **153 / 212** prompts got ≥1 valid sample (72.2%); 65/100 entities represented.
- Of covered prompts, 809 have all 3 samples valid, 23 have 2, 4 have 1 (remainder of attempts returned an empty response at `stop_reason="max_tokens"` — the reasoning-capable Sonnet model sometimes used the whole token budget before emitting YES/NO; raising `max_tokens` 8→20 mid-run reduced but did not eliminate this).
- Total requests attempted: 2610 (ok 2477, errored 133, mostly the empty-response issue above).

**All numbers below are for claude-sonnet-5 only, on this partial (~78%), randomly-subsampled
set of prompts. They are reported as provisional given the reduced and non-uniform-per-cell
sample sizes (n=70-96 entities per cell; see summary.csv for exact n per row).**

## FMR (false-match rate) and recall vs. k (profile prompts, 3x-sampled majority vote)

| query | k | n prompts | n entities | rate (maj. vote) | mean P(YES) | 95% entity-bootstrap CI | 3-of-3 confident share |
|---|---|---|---|---|---|---|---|
| negA | 1 | 70 | 70 | 0.000 | 0.007 | [0.000, 0.000] | 0.000 (n=68) |
| negA | 4 | 75 | 75 | 0.213 | 0.218 | [0.120, 0.307] | 0.213 (n=75) |
| negA | 16 | 76 | 76 | 0.263 | 0.263 | [0.171, 0.368] | 0.253 (n=75) |
| negB | 1 | 80 | 80 | 0.013 | 0.013 | [0.000, 0.037] | 0.013 (n=79) |
| negB | 4 | 75 | 75 | 0.107 | 0.107 | [0.040, 0.173] | 0.097 (n=72) |
| negB | 16 | 73 | 73 | 0.178 | 0.178 | [0.096, 0.274] | 0.178 (n=73) |
| negA+negB | 1 | 150 | 94 | 0.007 | 0.010 | [0.000, 0.020] | 0.007 (n=147) |
| negA+negB | 4 | 150 | 94 | 0.160 | 0.162 | [0.099, 0.231] | 0.156 (n=147) |
| negA+negB | 16 | 149 | 96 | 0.221 | 0.221 | [0.149, 0.302] | 0.216 (n=148) |
| pos | 1 | 77 | 77 | 0.130 | 0.134 | [0.065, 0.208] | 0.139 (n=72) |
| pos | 4 | 75 | 75 | 0.387 | 0.387 | [0.280, 0.493] | 0.397 (n=73) |
| pos | 16 | 82 | 82 | 0.732 | 0.728 | [0.634, 0.817] | 0.733 (n=75) |

FMR (negA+negB combined) rises from 0.7% at k=1 to 22.1% at k=16 — the same
accretion-bias direction found with the open-weight models, with claude-sonnet-5's absolute FMR
at k=16 (22.1%) inside the range reported for the open models. Recall also rises sharply
with k (13.0% → 73.2%), so the model is not simply becoming
more YES-biased overall — larger profiles sharpen both correct accretion (recall) and false
accretion (FMR).

## McNemar test, k=1 vs k=16 (paired by entity x query, majority vote; exact binomial on discordant pairs)

| query | n pairs | NO→YES (k1→k16) | YES→NO | same YES | same NO | McNemar p |
|---|---|---|---|---|---|---|
| negA | 54 | 12 | 0 | 0 | 42 | 4.88e-04 |
| negB | 60 | 10 | 0 | 0 | 50 | 1.95e-03 |
| negA+negB | 114 | 22 | 0 | 0 | 92 | 4.77e-07 |
| pos | 61 | 37 | 0 | 9 | 15 | 1.46e-11 |

Every discordant pair moves NO→YES as k grows from 1 to 16 (zero YES→NO reversals in any query
group), so the k1-vs-k16 shift is significant for negA (4.9e-04), negB (2.0e-03),
negA+negB combined (4.8e-07) and pos (1.5e-11).

## Snowball effect: clean vs. contaminated k=4 profile, second record of the intruder's entity

Paired by (entity, query): n=54 query-level pairs across 44 entities (65/100 entities
have ≥1 covered snowball pair; coverage is the random ~72% subsample described above, not all
100 entities' snowball prompts were sampled before the ceiling hit).

- Clean-profile acceptance of the intruder's 2nd record: **18.5%**
- Contaminated-profile acceptance (same entity, same k, one verified member swapped for the intruder): **25.9%**
- Gap: **+7.4 points** (bootstrap 95% CI over entities: [-2.0, 20.0] points)
- Paired sign test on discordant query-pairs: 6 up / 2 down, exact binomial p=0.29
- Entity-level sign test (collapsing the two negatives per entity): 5 up / 2 down / 37 tied, p=0.45

The gap is in the same direction as every open-weight model (contamination raises acceptance of
the intruder's other records), but **it is not statistically significant at this sample size**
(n=54 pairs / 44 entities) — both the paired and entity-level tests, and the bootstrap CI, are
compatible with no effect. This track did not reach the full ~100-200 planned snowball prompts
(only 153/212 sampled) before the token cap stopped all further sampling, so the
snowball result for Claude should be treated as inconclusive rather than a confirmed null —
distinct from the open-weight-model snowball result, which used the full prompt set and reached
significance.

## How faithful / limitations

Only claude-sonnet-5 was sampled in this first pass; claude-haiku-5-5 has **zero** completed calls because the
2.0M-token cap of the sampling environment (shared across models) was exhausted by the Sonnet run. Coverage of the planned prompt set
is therefore partial and was determined by where a randomly-shuffled sampling order happened to
run out, not by any deliberate prioritization among entities or k/q cells — cell sizes range
n=70-96 entities out of the 100 selected, and the clean/contaminated snowball comparison in
particular has only 44/100 entities represented, which is why its effect is directionally
consistent with the open-weight models but not significant here. All rates use a 3x-repeated-
sampling majority vote; 27 of 836 covered prompts have fewer than 3 valid samples
(the rest are usable but less precise P(YES) estimates) because claude-sonnet-5 occasionally
exhausted a small `max_tokens` budget on internal reasoning before emitting YES/NO — a known
artifact of the model's extended-thinking behavior at very low token budgets, partially (not
fully) mitigated here by raising `max_tokens` from 8 to 20 mid-run. Because only one judge model
ran, no same-family judge/subject confound applies (as it would for an LLM-as-judge task), but
no closed-vs-closed model comparison is possible — the only comparison in this track is
closed-model claude-sonnet-5 against the open-weight models from the other tracks.
