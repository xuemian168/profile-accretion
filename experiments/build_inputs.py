"""Rebuild every model input from FOPPA v1.1.3.

python experiments/build_inputs.py --foppa <dir with Agents.csv, Names.csv, LotSuppliers.csv> --out data/raw

Writes items_all.json (266 entities: 16 member records, one held-out member, two hard negatives),
jobs_mitig.jsonl (mitigation and contamination prompts) and blocks_e2e.json (80 confusable blocks for
online linking). The random seeds and call order reproduce the files used in the paper byte for byte.
The first 100 entities were drawn for the pilot, the remaining 166 afterwards; both steps are kept.
"""
import argparse, json, random, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ap = argparse.ArgumentParser(); ap.add_argument("--foppa", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); F = a.foppa.rstrip("/") + "/"

ag = pd.read_csv(F + "Agents.csv", dtype=str); nm = pd.read_csv(F + "Names.csv", dtype=str)
ls = pd.read_csv(F + "LotSuppliers.csv", dtype=str)
sup = ag[ag.agentId.isin(set(ls.agentId))].copy()                      # agents that won at least one lot
sup = sup[sup.siret.str.fullmatch(r"\d{14}", na=False)].copy()          # valid 14-digit SIRET
sup["siren"] = sup.siret.str[:9]                                         # legal entity = first 9 digits
nv = nm.groupby("agentId").name.apply(lambda x: list(pd.unique(x)))     # raw name variants per agent
sup["variants"] = [v if isinstance(v, list) else [n] for v, n in zip(sup.agentId.map(nv), sup["name"])]
sz = sup.groupby("siren").size()
n_sup_all = int(ag.agentId.isin(set(ls.agentId)).sum())
stats = {"supplier_agents": n_sup_all, "with_valid_siret": int(len(sup)), "distinct_siret": int(sup.siret.nunique()),
         "distinct_siren": int(sup.siren.nunique()), "siren_ge17": int((sz >= 17).sum())}
sup = sup.reset_index(drop=True)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())).strip()
sup["nname"] = sup["name"].map(norm)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True)
X = vec.fit_transform(sup.nname)
siren_arr = sup.siren.values


def make_disp(rng):
    def disp(r):  # one random raw name variant + city + postcode
        v = r.variants[int(rng.integers(len(r.variants)))]
        loc = " ".join(x for x in [str(r.city) if pd.notna(r.city) else "", str(r.zipcode) if pd.notna(r.zipcode) else ""] if x).strip()
        return f"{norm(v)} | {loc or 'unknown'}"
    return disp


def build_items(entities, rng):
    disp = make_disp(rng); out = []
    for s in entities:
        idx = sup.index[sup.siren == s].to_numpy(); rng.shuffle(idx); mem = idx[:16]; pos = idx[16]
        def nn(i, excl):
            sims = (X[i] @ X.T).toarray().ravel(); sims[siren_arr == s] = -1
            for j in excl: sims[j] = -1
            j = int(sims.argmax()); return j, float(sims[j])
        negA, simA = nn(mem[0], [])                                      # nearest other entity to the founding member
        best = None
        for m in mem[8:16]:                                              # nearest other entity to a later member
            j, sim = nn(m, [negA])
            if best is None or sim > best[1]: best = (j, sim, int(m))
        negB, simB, _ = best
        out.append({"siren": s, "members": [disp(sup.loc[m]) for m in mem], "member_nnames": [sup.nname[m] for m in mem],
                    "queries": {"pos": {"text": disp(sup.loc[pos]), "label": 1, "nname": sup.nname[pos]},
                                "negA": {"text": disp(sup.loc[negA]), "label": 0, "nname": sup.nname[negA], "sim": simA, "neg_siren": siren_arr[negA]},
                                "negB": {"text": disp(sup.loc[negB]), "label": 0, "nname": sup.nname[negB], "sim": simB, "neg_siren": siren_arr[negB]}}})
    return out


big = sz[sz >= 17].index.tolist(); random.Random(3).shuffle(big)
items = build_items(big[:100], np.random.default_rng(7))                 # pilot entities
done = {it["siren"] for it in items}
rest = [s for s in sz[sz >= 17].index if s not in done]; random.Random(11).shuffle(rest)
items += build_items(rest, np.random.default_rng(17))
json.dump(items, open(f"{a.out}/items_all.json", "w"), ensure_ascii=False)

# mitigation and contamination prompts
KS = [1, 2, 4, 8, 16]
SKEPT = ("Before answering, check whether anything in the new record points to a DIFFERENT legal entity: a different distinctive "
         "name element, a different legal form or a named subsidiary/division of a group, or a location that suggests another company. "
         "Answer YES only if the new record matches the entity and nothing points elsewhere.")
rng2 = np.random.default_rng(23); disp2 = make_disp(rng2)
agents_by_siren = sup.groupby("siren").groups
jobs = []
for it in items:
    s = it["siren"]; M = it["members"]
    for q in ["pos", "negA", "negB"]:
        qt = it["queries"][q]["text"]
        for k in KS: jobs.append(dict(siren=s, q=q, kind="skeptical", x=k, members=M[:k], query=qt, extra=SKEPT))
        jobs.append(dict(siren=s, q=q, kind="bounded_last4", x=16, members=M[12:16], query=qt, extra=""))
        jobs.append(dict(siren=s, q=q, kind="anchor_plus3", x=16, members=[M[0]] + M[13:16], query=qt, extra=""))
    for q in ["negA", "negB"]:                                           # a second record of the intruder's entity
        ns = it["queries"][q]["neg_siren"]; intr = it["queries"][q]["text"]
        cands = [disp2(sup.loc[i]) for i in agents_by_siren.get(ns, [])]; cands = [c for c in cands if c != intr]
        if not cands: continue
        q2 = cands[int(rng2.integers(len(cands)))]
        for k in [4, 16]:
            jobs.append(dict(siren=s, q=q + "_2nd", kind="clean", x=k, members=M[:k], query=q2, extra=""))
            jobs.append(dict(siren=s, q=q + "_2nd", kind="contaminated", x=k, members=M[:k - 1] + [intr], query=q2, extra=""))
with open(f"{a.out}/jobs_mitig.jsonl", "w") as f:
    for j in jobs: f.write(json.dumps(j, ensure_ascii=False) + "\n")

# confusable blocks for online linking
rng3 = np.random.default_rng(31); disp3 = make_disp(rng3)
sname = sup.groupby("siren").nname.agg(lambda x: x.mode().iloc[0])
sv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True).fit(sname.values)
SX = sv.transform(sname.values); sidx = {s: i for i, s in enumerate(sname.index)}
blocks = []
for s in [it["siren"] for it in items[:80]]:
    sims = (SX[sidx[s]] @ SX.T).toarray().ravel(); sims[sidx[s]] = -1
    nb = [sname.index[j] for j in np.argsort(-sims)[:5] if sims[j] >= 0.5]
    recs = []
    for e, cap in [(s, 24)] + [(n, 8) for n in nb]:
        ix = list(agents_by_siren[e]); rng3.shuffle(ix)
        recs += [{"text": disp3(sup.loc[i]), "siren": e} for i in ix[:cap]]
    order = rng3.permutation(len(recs))
    blocks.append({"block": s, "neighbors": nb, "records": [recs[i] for i in order]})
json.dump(blocks, open(f"{a.out}/blocks_e2e.json", "w"), ensure_ascii=False)
stats.update({"entities": len(items), "mitigation_prompts": len(jobs), "blocks": len(blocks),
              "block_records": sum(len(b["records"]) for b in blocks)})
json.dump(stats, open(f"{a.out}/foppa_stats.json", "w"), indent=1)
print("entities", len(items), "mitigation prompts", len(jobs), "blocks", len(blocks), "records", sum(len(b["records"]) for b in blocks))
