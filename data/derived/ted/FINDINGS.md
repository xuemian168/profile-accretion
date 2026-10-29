# TED (Czech Republic + Bulgaria) second-dataset replication — findings

## Dataset construction

- Source: local TED contract-award-notice CSVs (`export_CAN_*`, DG GROW subset), years **2014-2017**
  (the only contract-award years available in the local subset; 2018-2019 CAN files are not present
  — the task's suggested 2014-2019 window was followed only as far as the data allow).
- Winner (WIN_*) records across 2014-2017 total 2,286,452 rows (all countries except FR, which was
  excluded to avoid FOPPA overlap).
- `WIN_NATIONALID` coverage was measured per country and cross-checked against each country's own
  checksum formula (data/derived/ted/coverage.csv). The two countries selected — **Czech Republic**
  (IČO, 8-digit, mod-11 check digit) and **Bulgaria** (EIK/Bulstat, 9-digit, mod-11 check digit) — have
  the best combination of coverage (35.0% / 34.3% of winner rows carry a non-empty national ID),
  checksum-valid rate (98.5% / 99.2% of non-empty IDs pass the national checksum), and entity pool size
  at the required profile depth. UK Companies House numbers and Polish NIP (the task's suggested
  examples) were checked but rejected: UK CRN coverage is only 2.2% of winner rows and has no public
  checksum to validate against; Polish NIP coverage is 1.5% and only 40% of the few non-empty IDs pass
  the mod-11 checksum, versus >98% for CZ/BG.
- A record is a distinct (name, street address, town, postcode) tuple; an entity is the checksum-valid,
  normalised national ID. **102 entities** (16 CZ + 86 BG) have >=17 distinct records, enough to run the
  full k in {1,2,4,8,16} nested-profile design at k_max=16 exactly as in the main FOPPA experiment (no
  fallback to the reduced k<=8 design was needed). Records are displayed as
  "NAME | STREET ADDRESS | POSTCODE TOWN | COUNTRY"; Cyrillic (BG) names are preserved by a
  unicode-aware normaliser used for the char 3-4-gram TF-IDF hard-negative search (the FOPPA
  ASCII-only regex would have blanked every Bulgarian name out — caught and fixed during this run).
- Hard negatives negA/negB: TF-IDF char 3-4-gram nearest *other-entity* record to the founding member
  and to a later member, respectively (same procedure as `build_inputs.py`), drawn from the full
  population of 17,267 distinct valid-ID CZ+BG winner records (not just the 102 profile-eligible
  entities). negA/negB similarity to the query entity's own name averaged 0.64 / 0.55 (cosine);
  22/102 negA and 6/102 negB queries happen to share the *exact* company name as a member (common
  generic or branch names) — the `fmr_distinct` vs `fmr_exactname` split below isolates this.
- System prompt and record format were adapted to state the CZ/BG registration-number definition
  explicitly (`experiments/ted/score_accretion_ted.py`, `experiments/ted/score_jobs_ted.py`); scoring
  mechanics (next-token P(YES) from the chat template, Qwen3 `enable_thinking=False`) are otherwise
  identical to the main experiment.
- 6,426 controlled-accretion prompts and 2,634 mitigation/contamination prompts per model (102 entities
  x 3 query types x [16 singles + 4 sets + 1 reversed-order set] = 6,426; mitigation/contamination jobs
  scale the same way as `build_inputs.py`, giving 2,634 instead of FOPPA's 6,690 because this dataset has
  102 entities instead of 266). All six models produced exactly these counts — no truncation.

## Controlled accretion (k=1 -> k=16): does NOT clearly replicate

Unlike the main FOPPA result, growing a profile from one correct record to sixteen does **not**
produce a consistent rise in the false-match rate (FMR) on hard, distinct-name negatives here.
Entity-level bootstrap (2,000 resamples, entities resampled) and an exact entity-level sign test
(binomial test on each entity's own negA/negB-averaged FMR at k=16 vs k=1 — entities with no change
are excluded as ties) give:

| model | FMR k=1 | FMR k=16 | Delta FMR [95% CI] | sign-test p | entities up/down (tied) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 23.0% | 13.7% | -9.3 [-14.2, -4.4] pp | 0.00027 | 1/16 (tied 85) |
| Qwen2.5-14B | 14.2% | 14.7% | +0.5 [-1.5, 2.9] pp | 1 | 3/2 (tied 97) |
| Qwen2.5-32B | 13.7% | 15.7% | +2.0 [-0.0, 4.4] pp | 0.22 | 5/1 (tied 96) |
| Qwen3-32B | 14.2% | 14.2% | +0.0 [-2.5, 2.5] pp | 1 | 3/3 (tied 96) |
| OLMo-2-32B | 15.7% | 14.2% | -1.5 [-3.4, 0.5] pp | 0.38 | 1/4 (tied 97) |
| Mistral-24B | 12.7% | 14.7% | +2.0 [-1.0, 4.9] pp | 0.29 | 6/2 (tied 94) |

Only **Qwen2.5-7B-Instruct** shows a statistically significant change, and it runs in the
**opposite** direction from the main paper: FMR *falls* by 9.3 pp (95% CI [-14.2, -4.4] pp,
entity-level sign-test p=0.00027, 16 of
17 non-tied entities moving down, 85 of the
102 entities tied at this resolution) as the profile grows from one record to sixteen. The other five
models show no significant drift either way (all 95% CIs include zero). Recall and AUC stay high and
roughly flat across k for every model (recall 77-99% at k=1, 89-95% at k=16; AUC 0.91-0.93 throughout)
— the models' raw discriminative ability on this dataset is intact; it is specifically the
profile-size-driven FMR drift documented on FOPPA that does not transfer here.

A second qualitative reversal: on FOPPA the whole-profile FMR at k=16 exceeds the "OR of the model's
own single-member judgments" null (superadditive accretion). On TED the opposite holds for all six
models — the OR-null FMR at k=16 (19.1-35.3%) is 3-25 pp *higher* than the actual whole-profile FMR
at k=16 (13.7-15.7%) in every case, i.e. showing the whole profile at once is more conservative than
the single most trigger-happy member judged alone, not less.

Exact-name negatives (where the hard negative happens to share the query entity's literal company
name) remain the dominant source of false matches, consistent with FOPPA: at k=16, FMR on exact-name
negatives is 78.6-85.7% versus only 3.4-4.6% on distinct-name negatives, across all six models.

## Snowball contamination: replicates for most models, but not uniformly

Splicing one intruder record into an otherwise-clean profile (k=4 or k=16) and then querying a
*second, held-out* record of that same intruder entity raises the false-match rate for most
models/profile sizes, with a near-one-directional McNemar test (clean vs. contaminated, paired over
the n=123 cases with a usable second record):

| model | clean k=4 | contam. k=4 | p (k=4) | clean k=16 | contam. k=16 | p (k=16) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 2.4% | 8.1% | 0.016 | 4.1% | 7.3% | 0.12 |
| Qwen2.5-14B | 2.4% | 39.8% | 2.8e-14 | 2.4% | 24.4% | 1.5e-08 |
| Qwen2.5-32B | 2.4% | 67.5% | 1.7e-24 | 5.7% | 24.4% | 2.4e-07 |
| Qwen3-32B | 2.4% | 61.8% | 2.1e-22 | 5.7% | 46.3% | 2.4e-14 |
| OLMo-2-32B | 3.3% | 31.7% | 5.8e-11 | 4.9% | 13.8% | 0.00098 |
| Mistral-24B | 3.3% | 69.9% | 4.1e-25 | 6.5% | 35.0% | 5.8e-11 |

11/12 of the clean-vs-contaminated comparisons (6 models x 2 profile sizes) are significant at
p<0.02 — the sole exception is **Qwen2.5-7B at k=16** (p=0.125, clean 4.1% vs contaminated 7.3%, only
4 up / 0 down transitions among 123 paired cases: directionally consistent but underpowered, not a
null result). 9/12 comparisons reach p<1e-6 (the three that do not: Qwen2.5-7B at k=4,
p=0.0156; Qwen2.5-7B at k=16, p=0.125; and OLMo-2-32B at k=16, p=9.8e-4). Every one of the 12
comparisons moves in the contamination-raises-FMR direction (0 or at most 1 entity moving the "safer"
way in any comparison). So the snowball mechanism replicates clearly for 5 of 6 models at both profile
sizes; for the sixth (Qwen2.5-7B) the effect is directionally present but only reaches significance at
k=4, not at k=16 — a single wrong record admitted into an otherwise-correct profile biases most of
these models toward matching an unrelated later record of the intruder's entity, transferring from
French SIREN-based FOPPA data to Czech/Bulgarian TED data with richer address fields and checksummed,
non-French registration numbers, though less uniformly across models than the headline FOPPA
contamination result.

## Mitigation prompting

The skeptical-prompt, last-4-members and founding+last-3 variants move FMR by only a few percentage
points relative to the whole-profile (k=16) baseline on TED, with no consistent direction across
models (data/derived/ted/mitigation.csv) — similar in magnitude to, but not a clear confirmation or
contradiction of, the main-paper mitigation results, given that the underlying k=1-to-16 accretion
effect being mitigated is itself not significant on TED for 5 of 6 models.

## How faithful / limitations

This is a provisional, single-country-pair replication, not a full re-run of the main experiment.
Award years are capped at 2014-2017 (no 2018-2019 CAN files in the local subset); the entity pool
(102, 16 CZ + 86 BG) is about 38% the size of FOPPA's 266, so per-model confidence intervals are wider
and the non-significant results above should not be read as proof of no effect, only as "not detected
at this sample size." WIN_NATIONALID coverage (~35%) means the entity pool is drawn from the subset of
TED winners whose registration number was actually reported, which may not be representative of all
TED suppliers. The Spanish-CIF control-letter rule used during the country screen (data/derived/ted/coverage.csv)
applies a simplified either-digit-or-letter check (Spain was not selected, so this does not affect the
final entities). Mitigation and contamination p-values above are uncorrected for the twelve
comparisons; applying a Bonferroni correction (threshold 0.05/12=4.2e-3) leaves 10/12
contamination comparisons significant (the two Qwen2.5-7B rows, p=0.0156 and p=0.125, fall below the
corrected threshold; all other 10 remain significant). The sole significant plain-accretion result
(Qwen2.5-7B, entity-level sign-test p=0.00027) survives a x6 (one test per model) Bonferroni
correction (p_adj=0.0016).
