"""Summarise the length control (data/raw/length/length_<model>.jsonl -> data/derived/length/summary.csv).
Conditions: set1 founding member; rep4 / rep16 founding member repeated 4 / 16 times; set16 16 distinct members.
Exact McNemar test of set1 vs rep16 and set1 vs set16 on the 532 hard negatives."""
import os
import pandas as pd
from scipy.stats import binomtest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); os.chdir(ROOT)
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B", "OLMo-2-0325-32B-Instruct", "Mistral-Small-24B-Instruct-2501"]
def mc(a, b):
    up, dn = int((~a & b).sum()), int((a & ~b).sum()); return up, dn, (binomtest(up, up + dn).pvalue if up + dn else 1.0)
rows = []
for m in MODELS:
    d = pd.read_json(f"data/raw/length/length_{m}.jsonl", lines=True, dtype={"siren": str}); d["cond"] = d.kind + d.x.astype(str)
    w = d.pivot_table(index=["siren", "q"], columns="cond", values="p_yes")
    N = w[w.index.get_level_values("q") != "pos"] > .5; P = w.xs("pos", level="q") > .5
    r = dict(model=m, n_neg=len(N), n_pos=len(P), mass_min=d.mass.min())
    for c in ["set1", "rep4", "rep16", "set16"]: r[f"fmr_{c}"] = N[c].mean(); r[f"rec_{c}"] = P[c].mean()
    r["rep16_up"], r["rep16_down"], r["rep16_p"] = mc(N.set1, N.rep16)
    r["set16_up"], r["set16_down"], r["set16_p"] = mc(N.set1, N.set16)
    rows.append(r)
pd.DataFrame(rows).to_csv("data/derived/length/summary.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
