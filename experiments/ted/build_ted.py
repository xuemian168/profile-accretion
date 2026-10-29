"""Build the TED (non-French) replication of items_all.json / jobs_mitig.jsonl.

Second-dataset replication of experiments/build_inputs.py, outside France, using TED
contract-award-notice winner (WIN_*) records for Czech Republic (ICO, 8-digit mod-11
check digit) and Bulgaria (EIK/Bulstat, 9-digit mod-11 check digit) -- the two non-FR
countries with the best combination of WIN_NATIONALID coverage (~35%), checksum-valid
rate (>98%), and number of entities with >=17 distinct winner records (16 CZ + 86 BG =
102), per data/derived/ted/coverage.csv. A "record" is a distinct (name, address, town,
postcode) tuple attached to a normalised national ID (= entity). Records are displayed
as "NAME | STREET ADDRESS | POSTCODE TOWN | COUNTRY".

python experiments/ted/build_ted.py --cache data/raw/ted/win_records_cz_bg.csv --out data/raw/ted

Mirrors build_inputs.py's random seeds and call order (shuffle entities, then draw 16
members + 1 held-out positive per entity, then TF-IDF hard negatives, then mitigation /
contamination jobs) so the two datasets are as comparable as possible.
"""
import argparse, json, random, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

pop = pd.read_csv(a.cache, dtype=str)
pop = pop.drop_duplicates(subset=["eid", "name", "address", "town", "postcode"]).reset_index(drop=True)

# unicode-aware normaliser: keeps letters/digits in any script (BG names are Cyrillic;
# the FOPPA ASCII-only regex would blank them out), replaces everything else with a space.
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w]", " ", str(s).upper(), flags=re.UNICODE)).strip()
pop["nname"] = pop["name"].map(norm)

sz = pop.groupby("eid").size()
stats = {"win_records_valid_id": int(len(pop)), "distinct_entities": int(pop.eid.nunique()),
         "entities_ge17": int((sz >= 17).sum()), "countries": sorted(pop.country.unique().tolist())}

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True)
X = vec.fit_transform(pop.nname)
eid_arr = pop.eid.values


def disp(r):  # NAME | STREET ADDRESS | POSTCODE TOWN | COUNTRY
    return f"{str(r['name']).strip()} | {str(r['address']).strip()} | {str(r['postcode']).strip()} {str(r['town']).strip()} | {r['country']}"


def build_items(entities, rng):
    out = []
    for s in entities:
        idx = pop.index[pop.eid == s].to_numpy(); rng.shuffle(idx); mem = idx[:16]; pos = idx[16]

        def nn(i, excl):
            sims = (X[i] @ X.T).toarray().ravel(); sims[eid_arr == s] = -1
            for j in excl: sims[j] = -1
            j = int(sims.argmax()); return j, float(sims[j])

        negA, simA = nn(mem[0], [])                                   # nearest other entity to the founding member
        best = None
        for m in mem[8:16]:                                           # nearest other entity to a later member
            j, sim = nn(m, [negA])
            if best is None or sim > best[1]: best = (j, sim, int(m))
        negB, simB, _ = best
        country = pop.loc[mem[0], "country"]
        out.append({"eid": s, "country": country, "members": [disp(pop.loc[m]) for m in mem],
                    "member_nnames": [pop.nname[m] for m in mem],
                    "queries": {"pos": {"text": disp(pop.loc[pos]), "label": 1, "nname": pop.nname[pos]},
                                "negA": {"text": disp(pop.loc[negA]), "label": 0, "nname": pop.nname[negA], "sim": simA, "neg_eid": eid_arr[negA]},
                                "negB": {"text": disp(pop.loc[negB]), "label": 0, "nname": pop.nname[negB], "sim": simB, "neg_eid": eid_arr[negB]}}})
    return out


big_cz = sz[(sz >= 17) & pop.groupby("eid").country.first().eq("CZ")].index.tolist()
big_bg = sz[(sz >= 17) & pop.groupby("eid").country.first().eq("BG")].index.tolist()
random.Random(3).shuffle(big_cz); random.Random(3).shuffle(big_bg)
items = build_items(big_cz, np.random.default_rng(7))                  # CZ first (mirrors FOPPA pilot-then-rest order)
items += build_items(big_bg, np.random.default_rng(17))
json.dump(items, open(f"{a.out}/items.json", "w"), ensure_ascii=False)

# mitigation and contamination prompts (same five variants + snowball swap as build_inputs.py)
KS = [1, 2, 4, 8, 16]
SKEPT = ("Before answering, check whether anything in the new record points to a DIFFERENT legal entity: a different distinctive "
         "name element, a different legal form or a named subsidiary/division of a group, or a location that suggests another company. "
         "Answer YES only if the new record matches the entity and nothing points elsewhere.")
rng2 = np.random.default_rng(23)
agents_by_eid = pop.groupby("eid").groups
jobs = []
for it in items:
    s = it["eid"]; M = it["members"]
    for q in ["pos", "negA", "negB"]:
        qt = it["queries"][q]["text"]
        for k in KS: jobs.append(dict(eid=s, q=q, kind="skeptical", x=k, members=M[:k], query=qt, extra=SKEPT))
        jobs.append(dict(eid=s, q=q, kind="bounded_last4", x=16, members=M[12:16], query=qt, extra=""))
        jobs.append(dict(eid=s, q=q, kind="anchor_plus3", x=16, members=[M[0]] + M[13:16], query=qt, extra=""))
    for q in ["negA", "negB"]:                                         # a second record of the intruder's entity
        ns = it["queries"][q]["neg_eid"]; intr = it["queries"][q]["text"]
        cands = [disp(pop.loc[i]) for i in agents_by_eid.get(ns, [])]; cands = [c for c in cands if c != intr]
        if not cands: continue
        q2 = cands[int(rng2.integers(len(cands)))]
        for k in [4, 16]:
            jobs.append(dict(eid=s, q=q + "_2nd", kind="clean", x=k, members=M[:k], query=q2, extra=""))
            jobs.append(dict(eid=s, q=q + "_2nd", kind="contaminated", x=k, members=M[:k - 1] + [intr], query=q2, extra=""))
with open(f"{a.out}/jobs.jsonl", "w") as f:
    for j in jobs: f.write(json.dumps(j, ensure_ascii=False) + "\n")

stats.update({"entities": len(items), "entities_cz": len(big_cz), "entities_bg": len(big_bg), "mitigation_prompts": len(jobs)})
json.dump(stats, open(f"{a.out}/ted_stats.json", "w"), indent=1)
print("entities", len(items), "(cz", len(big_cz), "bg", len(big_bg), ") mitigation prompts", len(jobs))
