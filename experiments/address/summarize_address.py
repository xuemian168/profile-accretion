"""Address ablation summary: same entities/members/negatives as the main run, records with street address."""
import json, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B", "OLMo-2-0325-32B-Instruct", "Mistral-Small-24B-Instruct-2501"]
NEG = ["negA", "negB"]
items = {it["siren"]: it for it in json.load(open("data/raw/items_all.json"))}
nm = lambda t: t.split(" | ")[0]
exact = {(s, q): nm(it["queries"][q]["text"]) in {nm(m) for m in it["members"]} for s, it in items.items() for q in NEG}
ENT = sorted(items); rng = np.random.default_rng(2027); B = rng.integers(0, len(ENT), size=(2000, len(ENT)))
def tabs(d):
    S = d[d.kind == "single"].pivot_table(index=["siren", "q"], columns="x", values="p_yes")
    T = d[d.kind == "set"].pivot_table(index=["siren", "q"], columns="x", values="p_yes"); T[1] = S[0]; return S, T
rows = []
for cond, path in [("name+city", "data/raw/scores_{}.jsonl"), ("name+address+city", "data/raw/address/scores_{}.jsonl")]:
    for m in MODELS:
        d = pd.read_json(path.format(m), lines=True, dtype={"siren": str}); S, T = tabs(d)
        neg = {k: np.stack([T.loc[[(s, q) for s in ENT], k].values for q in NEG], 1) for k in (1, 16)}
        pos = {k: T.loc[[(s, "pos") for s in ENT], k].values for k in (1, 16)}
        ex = np.array([[exact[(s, q)] for q in NEG] for s in ENT])
        orr = np.stack([(S.loc[[(s, q) for s in ENT]].iloc[:, :16] > .5).any(axis=1).values for q in NEG], 1)
        f1, f16 = (neg[1] > .5).mean(1), (neg[16] > .5).mean(1)
        bs = [f16[i].mean() - f1[i].mean() for i in B]
        a, b = (neg[1] > .5).ravel(), (neg[16] > .5).ravel(); up, dn = int((~a & b).sum()), int((a & ~b).sum())
        auc = lambda k: roc_auc_score(np.r_[np.ones(len(ENT)), np.zeros(2 * len(ENT))], np.r_[pos[k], neg[k].ravel()])
        rows.append(dict(condition=cond, model=m, fmr1=(neg[1] > .5).mean(), fmr16=(neg[16] > .5).mean(), dfmr=np.mean(f16 - f1),
                         dfmr_lo=np.percentile(bs, 2.5), dfmr_hi=np.percentile(bs, 97.5), mc_up=up, mc_down=dn, mc_p=binomtest(up, up + dn).pvalue if up + dn else 1,
                         dist1=(neg[1][~ex] > .5).mean(), dist16=(neg[16][~ex] > .5).mean(), or16=orr.mean(),
                         rec1=(pos[1] > .5).mean(), rec16=(pos[16] > .5).mean(), auc1=auc(1), auc16=auc(16),
                         conf16=(neg[16] > .99).mean(), mass_min=d.mass.min()))
R = pd.DataFrame(rows); R.to_csv("data/derived/address/summary.csv", index=False)
pd.set_option("display.width", 250); print(R.drop(columns=["mc_up", "mc_down"]).round(3).to_string())
