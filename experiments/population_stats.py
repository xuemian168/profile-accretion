"""Describe the FOPPA supplier population relative to the 266 study entities.

python experiments/population_stats.py --foppa <dir with Agents.csv, LotSuppliers.csv> --out data/raw
Writes foppa_population.json (agents per SIREN distribution, lots won, coverage of the study sample).
"""
import argparse, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--foppa", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); F = a.foppa.rstrip("/") + "/"
ag = pd.read_csv(F + "Agents.csv", dtype=str); ls = pd.read_csv(F + "LotSuppliers.csv", dtype=str)
sup = ag[ag.agentId.isin(set(ls.agentId))].copy()
sup = sup[sup.siret.str.fullmatch(r"\d{14}", na=False)].copy(); sup["siren"] = sup.siret.str[:9]
sz = sup.groupby("siren").size()
lots = ls[ls.agentId.isin(set(sup.agentId))].merge(sup[["agentId", "siren"]], on="agentId").groupby("siren").size()
big = sz[sz >= 17].index
out = {"n_siren": int(len(sz)), "agents_per_siren_quantiles": {str(q): float(sz.quantile(q)) for q in [.5, .9, .99]},
       "share_siren_single_agent": float((sz == 1).mean()), "share_siren_ge2": float((sz >= 2).mean()),
       "n_siren_ge17": int(len(big)), "share_agents_in_ge17": float(sz[big].sum() / sz.sum()),
       "share_lots_in_ge17": float(lots.reindex(big).fillna(0).sum() / lots.sum()),
       "agents_ge17_quantiles": {str(q): float(sz[big].quantile(q)) for q in [0, .5, .9, 1]},
       "agents_per_entity": {s: int(sz[s]) for s in big}}
json.dump(out, open(a.out.rstrip("/") + "/foppa_population.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "agents_per_entity"})
