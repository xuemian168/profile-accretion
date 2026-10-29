# Systems track: extracted prompting strategies and re-implementation

Three published LLM-ER prompting systems are re-implemented against the same
FOPPA profiles (`data/raw/items_all.json`, 266 SIREN entities, k ∈ {1,4,16}
verified members, one fixed `pos`/`negA`/`negB` query per entity) used by the
main accretion experiment. Each is adapted minimally so that **only k varies**
while the query stays fixed, to probe whether the accretion effect (false
assignment of a hard negative rising with profile size) is specific to the
single whole-profile next-token prompt used in the main experiment or also
present in these published batching/selection/clustering formats.

## 1. BatchER (Fan, Han, Fan, Chai, Tang, Li, Du, *Cost-Effective In-Context
Learning for Entity Resolution: A Design Space Exploration*, ICDE 2024,
DOI 10.1109/ICDE60146.2024.00284)

**Published design (Sec. II).** BatchER formalizes an ER prompt as
`Desc ⊕ Demos ⊕ Question(s)`: a task description, a set of labeled
demonstration pairs, and a **batch of b pairwise match/non-match questions**
answered together in one completion, each question Qi about one record pair
(Ei1, Ei2) with model answer Ai ∈ {Yes, No}. The paper's design-space axes are
(i) *question batching* — how many pairwise questions go into one prompt —
and (ii) *demonstration selection* — fixed, random, or covering-based example
pairs. The paper's own experiments batch b pairwise questions drawn from
*different* candidate pairs produced by blocking; we repurpose the same
`Desc ⊕ Demos ⊕ Question(s)` **structural template** but make every question
in a batch a (profile-member, query) pair from the *same* entity, so k (batch
size) tracks the profile size exactly as in the main experiment.

**Our re-implementation.**
- `Desc`: same SIREN-matching task description as the main experiment's
  system prompt (`score_accretion.py`'s `SYS`), stated once per prompt.
- `Demos`: 2 **fixed** demonstration pairs (one match, one non-match),
  drawn once from an entity held out of all 266 query entities, reused
  identically across every job — the "Fixed" demonstration-selection
  category in BatchER Table I (the paper's simplest, selection-free
  baseline; we do not reimplement the covering-based selector, which needs
  a labeled candidate pool we do not have here).
- `Question(s)`: for profile size k, batch k pairwise questions
  Qi = "Record A: member_i | Record B: query — same legal entity? YES/NO",
  one query record and one profile member per sub-question, i = 1..k,
  asked in one prompt; the model must answer Q1..Qk in order.
- Scoring: greedy (temperature-0) generation (next-token YES/NO scoring
  does not apply once more than one answer must be produced per prompt);
  robust line-based parsing of "Qi: YES/NO"; parse failures counted and
  excluded from accuracy denominators, reported separately.
- Prediction: the entity-level call is the **OR** over the k sub-answers
  (query matches the profile if it matches *any* batched member) — this is
  the same aggregation used as the "any-alias OR" null baseline in the main
  pilot, letting us ask whether *batched* pairwise presentation departs from
  it as k grows, rather than re-deriving a new aggregation rule.

## 2. ComEM (Wang, Chen, Lin, Chen, Han, Sun, Wang, Zeng, *Match, Compare, or
Select? An Investigation of Large Language Models for Entity Matching*,
COLING 2025, ACL Anthology 2025.coling-main.8, pp. 96–109)

**Published design (Sec. 3, Table 5).** ComEM contrasts three interaction
strategies: (a) *matching* — independent pairwise Yes/No, no record
interaction; (b) *comparing* — pairwise *relative* judgments between two
candidates against one query; (c) *selecting* — present the query together
with its **n candidate matches as a numbered list plus an explicit
"none of the above"-style option**, and have the LLM pick one option in a
single completion, which is the format that lets LLM-ER incorporate global
consistency across candidates that pairwise matching ignores. The compound
ComEM pipeline chains matching/comparing (cheap filter) → selecting
(expensive, global) on the survivors; we only need the *selecting* primitive
since the accretion question is about global, candidate-set judgments.

**Our re-implementation (selecting strategy).** For a fixed query (pos /
negA / negB) and profile size k we build 3 candidates:
- Candidate **A**: k verified members of the *true* profile for this entity.
- Candidate **B**: k members of a different, unrelated entity from the same
  266-entity pool (index `(i+1) mod 266`), fixed per entity across all k and
  query types — a distractor cluster, since items_all.json does not give us
  member lists for `negA`/`negB`'s own SIREN, only their query text.
- Option **"none of the above"** (C).
The prompt lists A's and B's member records and asks the model to answer
with exactly one letter (A/B/C) identifying which cluster the query belongs
to, or "none". Correct answers: pos → A; negA/negB → C (the SIREN of the
hard negative is neither A nor B). Scoring: **next-token** softmax over the
3 option-letter ids (same technique as the main experiment), since the
output alphabet is fixed and small — this is the "prefer next-token scoring"
case the task asks for. False assignment = P(A) > P(B), P(C) at test time for
negA/negB; recall = P(A) argmax for pos.

## 3. LLM-CER (Fu, Tang, Khan, Mehrotra, Ke, Gao, *In-context Clustering-based
Entity Resolution with Large Language Models: A Design Space Exploration*,
Proc. ACM Manag. Data 3(4) [SIGMOD 2026], DOI 10.1145/3749170)

**Published design (Sec. 4, Fig. 3).** Instead of pairwise matching,
LLM-CER puts a **record set** of several entities' records in one prompt and
asks the LLM to directly **cluster** them — return the partition as a list
of groups of record indices — reducing the number of LLM calls from O(pairs)
to O(record-set). The paper's own design-space study varies set size,
diversity, variation and ordering; we vary **set composition by profile
size k**, holding the clustering-prompt format fixed.

**Our re-implementation.** Build one record set per (entity, query type, k):
k profile members (entity X) + the fixed query record + 2 fixed distractor
records drawn from a different entity Y (entity `(i+1) mod 266`, same two
records reused across k for comparability), each numbered 1..(k+3). Prompt:
"Group the following numbered records into clusters so that records in the
same cluster refer to the same real-world legal entity. Return your answer
as a list of clusters, each a list of record numbers." Scoring: greedy
generation; regex-parse the bracketed list(s) of integers into a partition.
Prediction: match if the query's record number is parsed into the same
cluster as a majority of the k profile-member record numbers; parse failures
(no valid bracketed partition covering all ids) are counted and reported
separately, not silently treated as non-match.

## Faithfulness summary (see also `data/derived/systems/FINDINGS.md`)

All three re-implementations keep the published strategy's *interaction
structure* (batched pairwise questions / candidate-set selection with a
"none" option / direct multi-record clustering) exactly, but substitute our
own domain (FOPPA supplier SIREN matching), our own fixed demonstrations
(BatchER) or distractor source (ComEM, LLM-CER — drawn from the same 266-
entity pool rather than the papers' blocking pipeline), and skip each
paper's cost-optimizing machinery (BatchER's covering-based demonstration
selector; ComEM's cheap-LLM pre-filter stage; LLM-CER's hierarchical
merge/misclustering-guardrail for record sets beyond one LLM call). They are
evaluated on the controlled k ∈ {1,4,16} accretion design, not the papers'
original benchmarks, so absolute F1/accuracy numbers are not comparable to
the original papers' reported results — only the *within-strategy* trend
across k is informative for the accretion question.
