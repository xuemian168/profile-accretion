"""Fit size-dependent thresholds directly to a target hard-negative FMR tau (unlike
experiments/fix/fit_thresholds.py, which pins the k=1 bin at t=0.5 and matches other bins to
whatever FMR that produces; here every bin, including k=1, is fit so its own FMR <= tau).

Fitting set: same 186 entities in items_all.json that are not a block id in blocks_e2e.json.
For each model, each k in {1,2,4,8,16}, and each tau in {0.05,0.10,0.15,0.20,0.30}: the smallest
threshold t such that the pooled hard-negative (negA+negB) FMR at k is <= tau; capped at the
largest attainable value below the maximum score if tau is unreachable.

python experiments/fix/sweep/fit_thresholds_tau.py --raw data/raw --out experiments/fix/sweep/thresholds_tau.json
"""
import argparse, json, os
import numpy as np
import pandas as pd

KS = [1, 2, 4, 8, 16]
TAUS = [0.05, 0.10, 0.15, 0.20, 0.30]
NEG = ["negA", "negB"]
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B",
          "OLMo-2-0325-32B-Instruct", "Mistral-Small-24B-Instruct-2501"]


def fit_threshold(neg_scores, target):
    v = np.sort(np.asarray(neg_scores, dtype=float))
    cand = np.unique(v)
    if len(cand) == 0:
        return 0.5, {"capped": False, "achieved_fmr": None, "note": "no data"}
    fm_at = np.array([(v > t).mean() for t in cand])
    ok = cand[(fm_at <= target) & (cand < v.max())]
    if len(ok):
        t = float(ok.min())
        return t, {"capped": False, "achieved_fmr": float((v > t).mean())}
    below_max = cand[cand < v.max()]
    t = float(below_max.max()) if len(below_max) else float(v.max())
    return t, {"capped": True, "achieved_fmr": float((v > t).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="experiments/fix/sweep/thresholds_tau.json")
    a = ap.parse_args()

    items = json.load(open(f"{a.raw}/items_all.json"))
    all_sirens = {it["siren"] for it in items}
    blocks = json.load(open(f"{a.raw}/blocks_e2e.json"))
    block_ids = {b["block"] for b in blocks}
    fit_sirens = sorted(all_sirens - block_ids)
    assert block_ids <= all_sirens
    print("fit entities:", len(fit_sirens), "excluded block ids:", len(block_ids))

    result = {"_meta": {"fit_entities": len(fit_sirens), "excluded_block_entities": len(block_ids),
                         "taus": TAUS, "ks": KS,
                         "note": "every bin (incl. k=1) is fit directly to its own tau target, unlike fit_thresholds.py"}}
    for m in MODELS:
        path = f"{a.raw}/scores_{m}.jsonl"
        if not os.path.exists(path):
            print("MISSING", path, "- skipping model"); continue
        d = pd.read_json(path, lines=True, dtype={"siren": str})
        d = d[d.siren.isin(fit_sirens)]
        result[m] = {}
        for tau in TAUS:
            bins = {}
            for k in KS:
                if k == 1:
                    negk = d[(d.kind == "single") & (d.x == 0) & (d.q.isin(NEG))].p_yes.values
                else:
                    negk = d[(d.kind == "set") & (d.x == k) & (d.q.isin(NEG))].p_yes.values
                assert len(negk) > 0, (m, k)
                t, info = fit_threshold(negk, tau)
                bins[str(k)] = {"t": t, "n_neg": int(len(negk)), **info}
            result[m][str(tau)] = {"bins": bins}
            print(m, "tau", tau, {k: round(v["t"], 4) for k, v in bins.items()})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
