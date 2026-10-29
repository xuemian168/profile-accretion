# Prompt-robustness variants (track: prompts)

All three variants keep a next-token YES/NO readout (P(YES) = softmax mass on
YES-family tokens / (YES-family + NO-family tokens), computed from the logits
at the position right after the point where the model is asked to answer).
They differ in instruction wording, record layout/answer format, and whether
the SIREN/legal-entity definition is given at all. The original prompt (used
for `data/raw/scores_<model>.jsonl` and `accretion_curves.csv`) is kept as the
baseline ("V0") for comparison; it is not re-run here.

## V0 — baseline (for reference, unchanged, already scored)

System:
> You are an entity-resolution expert for French public-procurement supplier
> records. Each record is 'NAME | CITY POSTCODE'. Two records refer to the
> same legal entity if they share the same SIREN (different
> establishments/branches of one company count as the SAME). Distinct legal
> entities in the same corporate group (subsidiaries, sister companies,
> franchisees) count as DIFFERENT.

User:
> Known entity — the following {k} record(s) were verified to belong to ONE
> legal entity:
> 1. {member_1}
> ...
> New record: {query}
>
> Does the new record belong to that same legal entity? Answer with exactly
> one word: YES or NO.

Readout: next-token P(YES) vs P(NO) right after the generation prompt.

## V1 — reworded instruction and record framing

Same underlying task definition (SIREN-based identity, subsidiaries/sister
companies/franchisees are distinct), paraphrased throughout, and the record
block is framed as a "cluster" under review by a "data-matching analyst"
rather than a "known entity".

System:
> You work as a data-matching analyst reconciling French public-procurement
> supplier listings. Records are formatted as 'NAME | CITY POSTCODE'. Two
> records denote the same legal entity exactly when they share an identical
> SIREN number; subsidiaries, sister firms, and franchisees that belong to
> the same corporate group are DIFFERENT legal entities even when the brand
> name matches.

User:
> A cluster of {k} confirmed record(s), all previously established as
> describing one single legal entity, is listed below:
> 1. {member_1}
> ...
> Candidate record under review: {query}
>
> Is the candidate record describing the same legal entity as the cluster
> above? Reply with a single word, YES or NO.

Readout: next-token P(YES) vs P(NO) right after the generation prompt
(identical mechanics to V0, only the text changed).

## V2 — numbered table + "Answer: YES/NO" format

Members are presented as a markdown table instead of a numbered list, and the
model is instructed to answer in the fixed format "Answer: YES" or
"Answer: NO". We prefill the assistant turn with the literal string
"Answer:" (appended verbatim after the model's own chat template, not parsed
as part of the template) and score the next token after that string, so the
readout is still a single forward pass over YES/NO-family tokens — just read
one token later in the response than V0/V1/V3.

System:
> You are an entity-resolution expert for French public-procurement supplier
> records. Each record is 'NAME | CITY POSTCODE'. Two records refer to the
> same legal entity if they share the same SIREN (different
> establishments/branches of one company count as the SAME). Distinct legal
> entities in the same corporate group (subsidiaries, sister companies,
> franchisees) count as DIFFERENT. Format your answer exactly as "Answer:
> YES" or "Answer: NO", with no other text.

User:
> Verified profile ({k} record(s) confirmed as one legal entity):
>
> | # | Record |
> |---|---|
> | 1 | {member_1} |
> ...
>
> Record to classify: {query}
>
> Does the record to classify belong to the same legal entity as the
> verified profile above?

Assistant (prefilled, not generated): `Answer:`

Readout: P(YES) vs P(NO) at the token immediately following the prefilled
"Answer:" string.

## V3 — minimal plain question, no SIREN/legal-entity definition

Drops the SIREN definition and the "legal entity" vocabulary entirely; asks
the bare question a non-expert annotator might be given, with a generic
system prompt that carries no entity-resolution instruction.

System:
> You are a helpful assistant.

User:
> These records are known to be about the same company:
> - {member_1}
> - ...
>
> Here is another record: {query}
>
> Do these records refer to the same company? Answer with exactly one word:
> YES or NO.

Readout: next-token P(YES) vs P(NO) right after the generation prompt
(same mechanics as V0/V1).

## Implementation note

All three variants are scored by `experiments/prompts/score_variants.py`
(`--variant {v1,v2,v3}`), adapted from `experiments/score_accretion.py`. To
keep the compute budget for this robustness check bounded, the job set per
variant per model is reduced relative to the original scoring run: for each
of the 266 entities we score k in {1, 4, 16} for all three queries (pos,
negA, negB) — not the full k in {1,2,4,8,16} grid or the reversed-order
control — plus the 16 single-member judgments for negA and negB only (needed
for the OR-null at k=16; single-member judgments for the positive query are
not needed for any reported statistic and are dropped). This is 41
prompts/entity x 266 entities = 10,906 prompts per model per variant
(vs. 16,758 in the original full-grid run), stated here as a declared
reduction in scope relative to the original protocol, not a silent cut.
