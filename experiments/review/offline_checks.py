"""Offline robustness checks, set 1 (no new model calls).

1. Negative A vs negative B accretion in the main FOPPA design (exact McNemar k=1 vs k=16 per negative).
2. Hardness of the selected negatives: TF-IDF cosine of each negative to the member it was chosen for (FOPPA vs TED).
3. Non-LLM reference: character 3-4-gram TF-IDF cosine between the displayed records the LLM sees
   (k=1 = query vs founding member, k=16 = maximum over the 16 members); 3b. FOPPA and TED accretion restricted
   to negatives with name cosine >= 0.7.
4. BCubed precision/recall/F1 for online linking (founding member, pairwise max, whole profile, and the
   cross-validated precision-matched profile operating point).
5. Score saturation in online linking for Qwen2.5-32B: share of join decisions at cluster size >= 8 whose best
   candidate has P(YES) >= 0.9999, for correct and incorrect best candidates.

python experiments/review/offline_checks.py
Outputs: data/derived/review/*.csv
"""
import argparse, glob, json, os, re
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); a = ap.parse_args()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); os.chdir(ROOT)
OUT = "data/derived/review"; os.makedirs(OUT, exist_ok=True)
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B",
          "OLMo-2-0325-32B-Instruct", "Mistral-Small-24B-Instruct-2501"]
items = json.load(open("data/raw/items_all.json"))

# 1. negative A / B --------------------------------------------------------------------------------
rows = []
for m in MODELS:
    d = pd.read_json(f"data/raw/scores_{m}.jsonl", lines=True, dtype={"siren": str})
    s1 = d[(d.kind == "single") & (d.x == 0)].set_index(["siren", "q"]).p_yes
    s16 = d[(d.kind == "set") & (d.x == 16)].set_index(["siren", "q"]).p_yes
    for q in ["negA", "negB"]:
        y1 = s1.xs(q, level="q") > .5; y16 = s16.xs(q, level="q").reindex(y1.index) > .5
        up = int((~y1 & y16).sum()); dn = int((y1 & ~y16).sum())
        rows.append(dict(model=m, neg=q, n=len(y1), fmr_k1=y1.mean(), fmr_k16=y16.mean(), up=up, down=dn,
                         p=binomtest(up, up + dn).pvalue if up + dn else 1.0))
pd.DataFrame(rows).to_csv(f"{OUT}/neg_ab.csv", index=False)

# 2. negative hardness -------------------------------------------------------------------------------
ted = json.load(open("data/raw/ted/items.json"))
hd = []
for name, its in [("FOPPA", items), ("TED", ted)]:
    for q in ["negA", "negB"]:
        v = np.array([it["queries"][q]["sim"] for it in its])
        hd.append(dict(dataset=name, neg=q, n=len(v), mean=v.mean(), median=np.median(v), q25=np.quantile(v, .25), q75=np.quantile(v, .75)))
pd.DataFrame(hd).to_csv(f"{OUT}/neg_hardness.csv", index=False)

# 3. TF-IDF baseline on what the LLM sees -----------------------------------------------------------------
# Vectoriser fit on all displayed records of the study (members, queries, online-linking blocks). "name" uses the
# displayed name only, "record" the full displayed record (name | city postcode).
blocks_ = json.load(open("data/raw/blocks_e2e.json"))
corpus = [m for it in items for m in it["members"]] + [it["queries"][q]["text"] for it in items for q in ["pos", "negA", "negB"]] \
    + [r["text"] for b in blocks_ for r in b["records"]]
base = []
for lab, f in [("name", lambda s: s.split(" | ")[0]), ("record", lambda s: s)]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True).fit([f(x) for x in corpus])
    bl = []
    for it in items:
        M = vec.transform([f(m) for m in it["members"]])
        for q in ["pos", "negA", "negB"]:
            s = (vec.transform([f(it["queries"][q]["text"])]) @ M.T).toarray().ravel()
            bl.append((int(q == "pos"), s[0], s.max()))
    B = pd.DataFrame(bl, columns=["y", "s1", "s16"])
    base.append(dict(input=lab, auc_k1=roc_auc_score(B.y, B.s1), auc_k16max=roc_auc_score(B.y, B.s16)))
pd.DataFrame(base).to_csv(f"{OUT}/tfidf_baseline.csv", index=False)

# 3b. TED and FOPPA restricted to name-similar negatives (cosine >= 0.7 to the member they were chosen for) ----
hs = []
for name, its, pat, key in [("FOPPA", items, "data/raw/scores_{}.jsonl", "siren"), ("TED", ted, "data/raw/ted/scores_{}.jsonl", "eid")]:
    sim = {(str(it[key]).lstrip("0"), q): it["queries"][q]["sim"] for it in its for q in ["negA", "negB"]}
    for m in MODELS:
        d = pd.read_json(pat.format(m), lines=True, dtype={key: str}); d[key] = d[key].str.lstrip("0")
        s1 = d[(d.kind == "single") & (d.x == 0) & (d.q != "pos")].set_index([key, "q"]).p_yes > .5
        s16 = d[(d.kind == "set") & (d.x == 16) & (d.q != "pos")].set_index([key, "q"]).p_yes.reindex(s1.index) > .5
        hi = pd.Series([sim[k] >= 0.7 for k in s1.index], index=s1.index)
        up = int((~s1[hi] & s16[hi]).sum()); dn = int((s1[hi] & ~s16[hi]).sum())
        hs.append(dict(dataset=name, model=m, n=int(hi.sum()), fmr_k1=s1[hi].mean(), fmr_k16=s16[hi].mean(), up=up, down=dn,
                       p=binomtest(up, up + dn).pvalue if up + dn else 1.0))
pd.DataFrame(hs).to_csv(f"{OUT}/similar_negatives.csv", index=False)

# 4. BCubed ----------------------------------------------------------------------------------------------
def bcubed_block(g):
    cl = g.groupby("cluster").siren.agg(list); ent = Counter(g.siren); P = R = 0.0
    for mem in cl:
        c = Counter(mem)
        for e, n in c.items():
            P += n * n / len(mem); R += n * n / ent[e]
    return P, R, len(g)

def bc(P, R, N):
    p, r = P / N, R / N; return p, r, 2 * p * r / (p + r)

c2 = lambda n: n * (n - 1) // 2
def pair_counts(g):
    both = Counter(zip(g.cluster, g.siren)); tp = sum(c2(n) for n in both.values())
    return tp, sum(c2(n) for n in Counter(g.cluster).values()) - tp, sum(c2(n) for n in Counter(g.siren).values()) - tp

rows = []
for m in MODELS:
    d = pd.read_json(f"data/raw/e2e_{m}.jsonl", lines=True, dtype={"siren": str, "block": str})
    for rep, g in d.groupby("rep"):
        P, R, N = np.array([bcubed_block(x) for _, x in g.groupby("block")]).sum(0)
        p, r, f = bc(P, R, N); rows.append(dict(model=m, rep=rep, bc_precision=p, bc_recall=r, bc_f1=f))
    # cross-validated precision-matched profile point (selection on pairwise precision, as in cv_operating_point.py)
    s = pd.read_json(f"data/raw/fix/sweep/sweep_{m}.jsonl", lines=True, dtype={"siren": str, "block": str})
    s = s[s.variant != "profile_audit_soft"]
    blocks = sorted(s.block.unique()); OPS = [v for v in s.variant.unique() if v.startswith("profile")]
    PC = {v: np.array([pair_counts(x) for _, x in g.groupby("block")], float) for v, g in s.groupby("variant")}
    BCB = {v: np.array([bcubed_block(x) for _, x in g.groupby("block")], float) for v, g in s.groupby("variant")}
    prec = lambda t: t[0] / (t[0] + t[1]) if t[0] + t[1] else 0.0
    rec = lambda t: t[0] / (t[0] + t[2])
    rng = np.random.default_rng(7); res = []
    for rep_ in range(20):
        folds = np.array_split(rng.permutation(len(blocks)), 5); sel = np.zeros((len(blocks), 3))
        for te in folds:
            tr = np.setdiff1d(np.arange(len(blocks)), te); pa = prec(PC["anchor_t0.5"][tr].sum(0))
            cands = [(rec(PC[v][tr].sum(0)), prec(PC[v][tr].sum(0)), v) for v in OPS]
            ok = [c for c in cands if c[1] >= pa]
            v = max(ok)[2] if ok else max(cands, key=lambda c: c[1])[2]
            sel[te] = BCB[v][te]
        res.append(bc(*sel.sum(0)))
    p, r, f = np.mean(res, axis=0); rows.append(dict(model=m, rep="cv_profile_point", bc_precision=p, bc_recall=r, bc_f1=f))
pd.DataFrame(rows).to_csv(f"{OUT}/bcubed.csv", index=False)

# 5. saturation in online linking ------------------------------------------------------------------------
sat = []
for m in MODELS:
    d = pd.read_json(f"data/raw/e2e_{m}.jsonl", lines=True, dtype={"siren": str, "block": str})
    d = d[d.rep == "profile"].sort_values(["block", "i"])
    founder, rec_ = {}, []
    for r in d.itertuples():
        # best candidate cluster id = argmax of cand_scores
        if r.cand_scores:
            best = max(r.cand_scores, key=r.cand_scores.get); key = (r.block, int(best))
            rec_.append(dict(size=r.size_before, p=r.p_best, correct=founder.get(key) == r.siren))
        if not r.joined: founder[(r.block, r.cluster)] = r.siren
    x = pd.DataFrame(rec_); big = x[x["size"] >= 8]
    sat.append(dict(model=m, n_decisions_size_ge8=len(big), share_saturated=(big.p >= .9999).mean(),
                    sat_correct=(big[big.correct].p >= .9999).mean(), sat_incorrect=(big[~big.correct].p >= .9999).mean(),
                    n_incorrect=int((~big.correct).sum())))
pd.DataFrame(sat).to_csv(f"{OUT}/saturation_online.csv", index=False)
print("done")
