"""Summarise the closed-model (Claude) subset: 100 FOPPA entities, k in {1,4,16}, 3 samples per prompt.

Sonnet: claude-sonnet-5.jsonl + claude-sonnet-5_part2.jsonl (default reasoning settings) and, as claude-sonnet-5-nothink,
sonnet_nothink_shardA/B.jsonl (thinking disabled, matching the Haiku configuration).
Haiku: haiku_shardA.jsonl + haiku_shardB2.jsonl (thinking disabled). haiku_shardB.jsonl (thinking on, partial) is excluded.
Writes data/derived/closed/summary_full.csv and snowball_full.csv.
"""
import json, numpy as np, pandas as pd
from scipy.stats import binomtest
R, D = "data/raw/closed", "data/derived/closed"
P = pd.DataFrame([json.loads(l) for l in open(f"{R}/prompts.jsonl")])[["pid", "siren", "q", "kind", "k"]]
def load(files):
    r = pd.concat([pd.read_json(f"{R}/{f}", lines=True, dtype={"siren": str}) for f in files])
    r = r[r.ans.notna() & r.err.isna()].drop_duplicates(["pid", "sample"], keep="last")
    return r
models = {"claude-sonnet-5": ["claude-sonnet-5.jsonl", "claude-sonnet-5_part2.jsonl"], "claude-sonnet-5-nothink": ["sonnet_nothink_shardA.jsonl", "sonnet_nothink_shardB.jsonl"], "claude-haiku-5-5": ["haiku_shardA.jsonl", "haiku_shardB2.jsonl"]}
rng = np.random.default_rng(2027); rows, snow = [], []
for m, fs in models.items():
    r = load(fs); g = r.groupby("pid").ans.agg(["mean", "count"]).reset_index()
    d = P.merge(g, on="pid", how="left"); d["yes"] = d["mean"] > .5; d["conf"] = (d["mean"] == 1) & (d["count"] >= 3)
    s = d[d.kind == "set"]; ents = sorted(s.siren.unique())
    def rate(q, k, col="yes"):
        x = s[(s.q.isin(q)) & (s.k == k) & s["count"].notna()]; return x[col].mean(), len(x)
    for k in (1, 4, 16):
        f, nf = rate(["negA", "negB"], k); rc, nr = rate(["pos"], k); cf, _ = rate(["negA", "negB"], k, "conf")
        rows.append(dict(model=m, k=k, fmr=f, n_neg=nf, recall=rc, n_pos=nr, conf_fm=cf, mean_p_neg=s[(s.q.isin(["negA","negB"])) & (s.k == k)]["mean"].mean()))
    # paired k1 vs k16 on negatives with both observed
    w = s[s.q.isin(["negA", "negB"])].pivot_table(index=["siren", "q"], columns="k", values="yes", aggfunc="first").dropna(subset=[1, 16])
    up = int(((~w[1].astype(bool)) & w[16].astype(bool)).sum()); dn = int((w[1].astype(bool) & ~w[16].astype(bool)).sum())
    pe = w.groupby(level=0).apply(lambda x: x[16].mean() - x[1].mean()); E = pe.index.values
    bs = [pe.reindex(rng.choice(E, len(E))).mean() for _ in range(2000)]
    rows[-1].update(mc_up=up, mc_down=dn, mc_p=binomtest(up, up + dn).pvalue if up + dn else 1, dfmr_paired=pe.mean(), dfmr_lo=np.percentile(bs, 2.5), dfmr_hi=np.percentile(bs, 97.5), n_paired=len(w))
    sn = d[d.kind.isin(["clean", "contaminated"])].pivot_table(index=["siren", "q"], columns="kind", values="yes", aggfunc="first").dropna()
    a, b = sn.clean.astype(bool), sn.contaminated.astype(bool); up, dn = int((~a & b).sum()), int((a & ~b).sum())
    snow.append(dict(model=m, n_pairs=len(sn), clean=a.mean(), contaminated=b.mean(), up=up, down=dn, p=binomtest(up, up + dn).pvalue if up + dn else 1))
S = pd.DataFrame(rows); N = pd.DataFrame(snow)
S.to_csv(f"{D}/summary_full.csv", index=False); N.to_csv(f"{D}/snowball_full.csv", index=False)
pd.set_option("display.width", 250); print(S.round(4).to_string()); print(N.round(4).to_string())
