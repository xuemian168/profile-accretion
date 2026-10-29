"""Re-analyse the main controlled design with the log-odds readout (data/raw/logodds/lo_<model>.jsonl).

For each model: agreement of the new p_yes with the original run, share of k=16 scores with p_yes == 1 and the
number of distinct log-odds values among them, AUC at k=1 / k=16 on log-odds, and the cross-validated matched
recall of build_assets.py (5 folds over entities x 20 repetitions; on the training entities the k=16 threshold is
the smallest log-odds value whose false-match rate does not exceed the k=1 rate at P(YES) > 0.5).
Output: data/derived/logodds/summary.csv
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); os.chdir(ROOT)
os.makedirs("data/derived/logodds", exist_ok=True)
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B"]
rows = []
for m in MODELS:
    f = f"data/raw/logodds/lo_{m}.jsonl"
    if not os.path.exists(f): continue
    d = pd.read_json(f, lines=True, dtype={"siren": str}); o = pd.read_json(f"data/raw/scores_{m}.jsonl", lines=True, dtype={"siren": str})
    key = ["siren", "q", "kind", "x"]; j = d.merge(o[key + ["p_yes"]], on=key, suffixes=("", "_orig"))
    agree = ((j.p_yes > .5) == (j.p_yes_orig > .5)).mean()
    s1 = d[(d.kind == "single") & (d.x == 0)].set_index(["siren", "q"]); s16 = d[(d.kind == "set") & (d.x == 16)].set_index(["siren", "q"])
    lab = lambda s: (s.index.get_level_values("q") == "pos").astype(int)
    sat = s16[s16.p_yes >= 1.0]
    ENT = sorted(s1.index.get_level_values("siren").unique())
    neg1 = np.array([[s1.loc[(e, q), "lo"] for q in ("negA", "negB")] for e in ENT]); neg16 = np.array([[s16.loc[(e, q), "lo"] for q in ("negA", "negB")] for e in ENT])
    pos1 = np.array([s1.loc[(e, "pos"), "lo"] for e in ENT]); pos16 = np.array([s16.loc[(e, "pos"), "lo"] for e in ENT])
    rng = np.random.default_rng(99); res = []
    for rep in range(20):
        for fo in np.array_split(rng.permutation(len(ENT)), 5):
            tr = np.setdiff1d(np.arange(len(ENT)), fo); target = (neg1[tr] > 0).mean(); v = np.sort(neg16[tr].ravel())
            cand = np.unique(v); fm_at = np.array([(v > t).mean() for t in cand]); ok = cand[(fm_at <= target) & (cand < v.max())]
            if len(ok) == 0: res.append(dict(attain=False)); continue
            t = ok.min(); res.append(dict(attain=True, fmr16=(neg16[fo] > t).mean(), rec16=(pos16[fo] > t).mean(), fmr1=(neg1[fo] > 0).mean(), rec1=(pos1[fo] > 0).mean()))
    Rr = pd.DataFrame(res); A = Rr[Rr.attain] if Rr.attain.any() else None
    rows.append(dict(model=m, n=len(d), agree_with_orig=agree, sat16=len(sat), sat16_share=len(sat) / len(s16), sat16_distinct_lo=sat.lo.round(3).nunique(),
                     sat16_lo_min=sat.lo.min() if len(sat) else np.nan, sat16_lo_max=sat.lo.max() if len(sat) else np.nan,
                     auc1_lo=roc_auc_score(lab(s1[s1.index.get_level_values("siren").isin(ENT)].loc[pd.MultiIndex.from_product([ENT, ["pos", "negA", "negB"]])]),
                                           s1.loc[pd.MultiIndex.from_product([ENT, ["pos", "negA", "negB"]])].lo),
                     auc16_lo=roc_auc_score(lab(s16.loc[pd.MultiIndex.from_product([ENT, ["pos", "negA", "negB"]])]), s16.loc[pd.MultiIndex.from_product([ENT, ["pos", "negA", "negB"]])].lo),
                     auc16_p=roc_auc_score(lab(s16.loc[pd.MultiIndex.from_product([ENT, ["pos", "negA", "negB"]])]), s16.loc[pd.MultiIndex.from_product([ENT, ["pos", "negA", "negB"]])].p_yes),
                     attain=Rr.attain.mean(), cv_fmr1=A.fmr1.mean() if A is not None else np.nan, cv_rec1=A.rec1.mean() if A is not None else np.nan,
                     cv_fmr16=A.fmr16.mean() if A is not None else np.nan, cv_rec16=A.rec16.mean() if A is not None else np.nan))
pd.DataFrame(rows).to_csv("data/derived/logodds/summary.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
