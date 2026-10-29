"""Offline robustness checks, set 2 (no new model calls).

1. LLM calls per record in online linking for founding member, pairwise max and whole profile
   (replaying cluster sizes from data/raw/e2e_<model>.jsonl; pairwise max scores every shown member of each candidate).
2. Holm correction across all model-level McNemar tests (k=1 vs k=16, pooled hard negatives) behind Table III.
3. TED results by country (descriptive).
4. Parse failures of the LLM-CER-style format by query type and k.
5. Confident false matches at P(YES) > 0.95 / 0.99 / 0.999 (best single member vs 16-member profile).
Outputs: data/derived/review/{calls_online,holm_table3,ted_by_country,llmcer_parse,conf_thresholds}.csv
"""
import json, os, re
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import binomtest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); os.chdir(ROOT)
OUT = "data/derived/review"; os.makedirs(OUT, exist_ok=True)
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B", "OLMo-2-0325-32B-Instruct", "Mistral-Small-24B-Instruct-2501"]
MAXM = 16

# 1. calls per record ------------------------------------------------------------------------------------
rows = []
for m in MODELS:
    d = pd.read_json(f"data/raw/e2e_{m}.jsonl", lines=True, dtype={"siren": str, "block": str})
    for rep, g in d.groupby("rep"):
        calls = 0
        for _, gb in g.sort_values(["block", "i"]).groupby("block"):
            size = {}
            for r in gb.itertuples():
                cands = [int(c) for c in r.cand_scores]
                calls += sum(min(size[c], MAXM) for c in cands) if rep == "pairmax" else len(cands)
                size[r.cluster] = size.get(r.cluster, 0) + 1
        rows.append(dict(model=m, rep=rep, records=len(g), calls=calls, calls_per_record=calls / len(g)))
pd.DataFrame(rows).to_csv(f"{OUT}/calls_online.csv", index=False)

# 2. Holm across Table III McNemar tests -------------------------------------------------------------------
def mc(y1, y16):
    up, dn = int((~y1 & y16).sum()), int((y1 & ~y16).sum()); return up, dn, (binomtest(up, up + dn).pvalue if up + dn else 1.0)
T = []
for m in MODELS:
    d = pd.read_json(f"data/raw/scores_{m}.jsonl", lines=True, dtype={"siren": str})
    a = d[(d.kind == "single") & (d.x == 0) & (d.q != "pos")].set_index(["siren", "q"]).p_yes > .5
    b = d[(d.kind == "set") & (d.x == 16) & (d.q != "pos")].set_index(["siren", "q"]).p_yes.reindex(a.index) > .5
    T.append(("FOPPA main", m, *mc(a, b)))
    d = pd.read_json(f"data/raw/ted/scores_{m}.jsonl", lines=True, dtype={"eid": str})
    a = d[(d.kind == "single") & (d.x == 0) & (d.q != "pos")].set_index(["eid", "q"]).p_yes > .5
    b = d[(d.kind == "set") & (d.x == 16) & (d.q != "pos")].set_index(["eid", "q"]).p_yes.reindex(a.index) > .5
    T.append(("TED", m, *mc(a, b)))
AD = pd.read_csv("data/derived/address/summary.csv"); AD = AD[AD.condition == "name+address+city"]
for r in AD.itertuples(): T.append(("address", r.model, int(r.mc_up), int(r.mc_down), r.mc_p))
PV = pd.read_csv("data/derived/prompts/summary.csv")
for r in PV.itertuples(): T.append((f"prompt {r.variant}", r.model, int(r.mcnemar_up_k1v16), int(r.mcnemar_down_k1v16), r.mcnemar_p_k1v16))
SY = pd.read_csv("data/derived/systems/summary.csv"); SY = SY[(SY.k == 16) & SY.qtype.isin(["negA", "negB"])]
for (s, m), g in SY.groupby(["strategy", "model"]):
    up, dn = int(g.discordant_0to1.sum()), int(g.discordant_1to0.sum())
    T.append((s, m, up, dn, binomtest(up, up + dn).pvalue if up + dn else 1.0))
CL = pd.read_csv("data/derived/closed/summary_full.csv"); CL = CL[CL.k == 16]
for r in CL.itertuples(): T.append(("closed", r.model, int(r.mc_up), int(r.mc_down), r.mc_p))
H = pd.DataFrame(T, columns=["setting", "model", "up", "down", "p"]).sort_values("p").reset_index(drop=True)
n = len(H); adj = np.maximum.accumulate([(n - i) * p for i, p in enumerate(H.p)]); H["p_holm"] = np.minimum(1, adj)
H["sig_up_raw"] = (H.p < .05) & (H.up > H.down); H["sig_up_holm"] = (H.p_holm < .05) & (H.up > H.down)
H.to_csv(f"{OUT}/holm_table3.csv", index=False)

# 3. TED by country -----------------------------------------------------------------------------------------
ted = json.load(open("data/raw/ted/items.json")); country = {str(it["eid"]).lstrip("0"): it["country"] for it in ted}
rows = []
for m in MODELS:
    d = pd.read_json(f"data/raw/ted/scores_{m}.jsonl", lines=True, dtype={"eid": str}); d["eid"] = d.eid.str.lstrip("0")
    a = d[(d.kind == "single") & (d.x == 0) & (d.q != "pos")].set_index(["eid", "q"]).p_yes > .5
    b = d[(d.kind == "set") & (d.x == 16) & (d.q != "pos")].set_index(["eid", "q"]).p_yes.reindex(a.index) > .5
    pa = d[(d.kind == "single") & (d.x == 0) & (d.q == "pos")].set_index("eid").p_yes > .5
    cc = pd.Series([country[e] for e, _ in a.index], index=a.index)
    for c in ["CZ", "BG"]:
        up, dn, p = mc(a[cc == c], b[cc == c])
        rows.append(dict(model=m, country=c, n_neg=int((cc == c).sum()), fmr_k1=a[cc == c].mean(), fmr_k16=b[cc == c].mean(), up=up, down=dn, p=p,
                         rec_k1=pa[[country[e] == c for e in pa.index]].mean()))
pd.DataFrame(rows).to_csv(f"{OUT}/ted_by_country.csv", index=False)

# 4. LLM-CER parse failures -----------------------------------------------------------------------------------
SA = pd.read_csv("data/derived/systems/summary.csv"); SA = SA[SA.strategy.str.startswith("llmcer")]
SA[["model", "qtype", "k", "n", "parse_fail_rate", "rate"]].to_csv(f"{OUT}/llmcer_parse.csv", index=False)

# 5. confident false matches at several cut-offs ----------------------------------------------------------------
rows = []
for m in MODELS:
    d = pd.read_json(f"data/raw/scores_{m}.jsonl", lines=True, dtype={"siren": str}); n_ = d[d.q != "pos"]
    best = n_[n_.kind == "single"].groupby(["siren", "q"]).p_yes.max()
    prof = n_[(n_.kind == "set") & (n_.x == 16)].set_index(["siren", "q"]).p_yes
    for t in (.95, .99, .999):
        rows.append(dict(model=m, cut=t, best_member=(best > t).mean(), profile=(prof > t).mean()))
pd.DataFrame(rows).to_csv(f"{OUT}/conf_thresholds.csv", index=False)
print("done")
