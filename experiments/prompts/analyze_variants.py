"""Prompt-robustness analysis (track: prompts).

Reads data/raw/prompts/scores_{v1,v2,v3}_<model>.jsonl (produced by
score_variants.py on Aqua) plus the original-prompt baseline
data/derived/accretion_curves.csv (V0), and writes:
  data/derived/prompts/summary.csv                   -- one row per (model, variant)
  data/derived/prompts/accretion_curves_by_variant.csv -- one row per (model, variant, k)
  data/derived/prompts/mcnemar_k1v16.csv
  data/derived/prompts/bootstrap_entity_by_variant.csv
  data/derived/prompts/yesno_mass_by_variant.csv

python experiments/prompts/analyze_variants.py   (run from the repository root)
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

RAW = "data/raw"
PROMPTS_RAW = f"{RAW}/prompts"
DER = "data/derived/prompts"
os.makedirs(DER, exist_ok=True)

NEG = ["negA", "negB"]
KS = [1, 4, 16]
B = 2000  # bootstrap resamples

MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B",
          "OLMo-2-0325-32B-Instruct", "Mistral-Small-24B-Instruct-2501"]
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen2.5-14B-Instruct": "Qwen2.5-14B", "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
         "Qwen3-32B": "Qwen3-32B", "OLMo-2-0325-32B-Instruct": "OLMo-2-32B", "Mistral-Small-24B-Instruct-2501": "Mistral-24B"}
VARIANTS = ["v1", "v2", "v3"]

items = {it["siren"]: it for it in json.load(open(f"{RAW}/items_all.json"))}
ENT = sorted(items)
nm = lambda t: t.split(" | ")[0]
exact = {(s, q): nm(it["queries"][q]["text"]) in {nm(m) for m in it["members"]} for s, it in items.items() for q in NEG}


def split(df, qs):
    return df[df.index.get_level_values("q").isin(qs)]


def mcnemar(a, b):
    up = int(((~a) & b).sum()); dn = int((a & ~b).sum())
    return up, dn, (binomtest(up, up + dn).pvalue if up + dn else 1.0)


def load_variant(tag, variant):
    d = pd.read_json(f"{PROMPTS_RAW}/scores_{variant}_{tag}.jsonl", lines=True, dtype={"siren": str})
    assert len(d) == 10906, (tag, variant, len(d))
    S = d[d.kind == "single"].pivot_table(index=["siren", "q"], columns="x", values="p_yes")  # negA/negB only, cols 0..15
    T = d[d.kind == "set"].pivot_table(index=["siren", "q"], columns="x", values="p_yes")     # pos/negA/negB, cols 1,4,16
    return S, T, d


rng_b = np.random.default_rng(2027)
boot_idx = rng_b.integers(0, len(ENT), size=(B, len(ENT)))
ci = lambda a: np.percentile(a, [2.5, 97.5])

curve_rows, mcnemar_rows, boot_rows, mass_rows = [], [], [], []
for tag in MODELS:
    for variant in VARIANTS:
        S, T, d = load_variant(tag, variant)
        mass_rows.append(dict(model=tag, variant=variant, mean_mass=d.mass.mean(), min_mass=d.mass.min()))
        Tn, Tp = split(T, NEG), split(T, ["pos"])
        ex = np.array([exact[i] for i in Tn.index])
        orr16 = (S.loc[Tn.index, list(range(16))] > .5).any(axis=1)

        for k in KS:
            yn = Tn[k] > .5
            auc = roc_auc_score(np.r_[np.ones(len(Tp)), np.zeros(len(Tn))], np.r_[Tp[k].values, Tn[k].values])
            row = dict(model=tag, variant=variant, k=k, fmr=yn.mean(), recall=(Tp[k] > .5).mean(), auc=auc,
                       fmr_distinct=yn[~ex].mean(), fmr_exactname=yn[ex].mean() if ex.any() else np.nan,
                       n_neg=len(Tn), n_pos=len(Tp))
            if k == 16:
                row["fmr_or"] = orr16.mean(); row["agree_or"] = (yn == orr16).mean()
                row["exceedance_pp"] = 100 * (row["fmr"] - row["fmr_or"])
                row["confident_fmr_p99"] = (Tn[16] > .99).mean()
            curve_rows.append(row)

        up, dn, p = mcnemar(Tn[1] > .5, Tn[16] > .5)
        mcnemar_rows.append(dict(model=tag, variant=variant, up=up, down=dn, p=p, n=up + dn))

        neg_k = {k: np.stack([T.loc[[(s, q) for s in ENT], k].values for q in NEG], 1) for k in KS}
        pos_k = {k: T.loc[[(s, "pos") for s in ENT], k].values for k in KS}
        f_ent = {k: (neg_k[k] > .5).mean(1) for k in KS}
        r_ent = {k: pos_k[k] > .5 for k in KS}
        bt = {f"fmr{k}": [] for k in KS}; bt.update({f"rec{k}": [] for k in KS}); bt.update({f"auc{k}": [] for k in KS})
        bt["dfmr"] = []; bt["drec"] = []; bt["dauc"] = []
        for ix in boot_idx:
            for k in KS:
                bt[f"fmr{k}"].append(f_ent[k][ix].mean())
                bt[f"rec{k}"].append(r_ent[k][ix].mean())
                y = np.r_[np.ones(len(ix)), np.zeros(2 * len(ix))]
                bt[f"auc{k}"].append(roc_auc_score(y, np.r_[pos_k[k][ix], neg_k[k][ix].ravel()]))
            bt["dfmr"].append(bt["fmr16"][-1] - bt["fmr1"][-1])
            bt["drec"].append(bt["rec16"][-1] - bt["rec1"][-1])
            bt["dauc"].append(bt["auc16"][-1] - bt["auc1"][-1])
        row = dict(model=tag, variant=variant)
        for name, vals in bt.items():
            lo, hi = ci(vals); row[f"{name}_mean"] = np.mean(vals); row[f"{name}_lo"] = lo; row[f"{name}_hi"] = hi
        boot_rows.append(row)

C = pd.DataFrame(curve_rows); MC = pd.DataFrame(mcnemar_rows); BT = pd.DataFrame(boot_rows); MASS = pd.DataFrame(mass_rows)

pivot_c = C.pivot_table(index=["model", "variant"], columns="k", values=["fmr", "recall", "auc"])
pivot_c.columns = [f"{a}_k{b}" for a, b in pivot_c.columns]
pivot_c = pivot_c.reset_index()
k16_extra = C[C.k == 16][["model", "variant", "fmr_or", "exceedance_pp", "confident_fmr_p99", "agree_or"]].rename(
    columns={"fmr_or": "fmr_or_k16", "agree_or": "agree_or_k16"})

summary = pivot_c.merge(k16_extra, on=["model", "variant"]).merge(
    MC.rename(columns={"up": "mcnemar_up_k1v16", "down": "mcnemar_down_k1v16", "p": "mcnemar_p_k1v16", "n": "mcnemar_n_k1v16"}),
    on=["model", "variant"]).merge(
    BT[["model", "variant", "dfmr_mean", "dfmr_lo", "dfmr_hi", "drec_mean", "drec_lo", "drec_hi", "dauc_mean", "dauc_lo", "dauc_hi"]],
    on=["model", "variant"]).merge(
    MASS.rename(columns={"mean_mass": "yesno_mass_mean", "min_mass": "yesno_mass_min"}), on=["model", "variant"])

V0 = pd.read_csv("data/derived/accretion_curves.csv")
V0 = V0[V0.k.isin(KS)][["model", "k", "fmr", "fmr_or", "recall", "auc"]]
v0p = V0.pivot_table(index="model", columns="k", values=["fmr", "fmr_or", "recall", "auc"])
v0p.columns = [f"v0_{a}_k{b}" for a, b in v0p.columns]
summary = summary.merge(v0p.reset_index(), on="model", how="left")
summary["short_model"] = summary.model.map(SHORT)
summary = summary.sort_values(["model", "variant"]).reset_index(drop=True)

summary.to_csv(f"{DER}/summary.csv", index=False)
C.to_csv(f"{DER}/accretion_curves_by_variant.csv", index=False)
MC.to_csv(f"{DER}/mcnemar_k1v16.csv", index=False)
BT.to_csv(f"{DER}/bootstrap_entity_by_variant.csv", index=False)
MASS.to_csv(f"{DER}/yesno_mass_by_variant.csv", index=False)
print("wrote", DER)
