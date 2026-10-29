"""Fit size-dependent decision thresholds for profile-size bins {1,2,4,8,16}.

Fitting set: the 186 entities in items_all.json that are NOT used as a block id in
blocks_e2e.json (i.e. never appear in the online-linking evaluation), so the threshold
fit and the end-to-end evaluation never touch the same entities.

For each model and nested k in {2,4,8,16}, pick the smallest threshold t_k such that the
pooled hard-negative (negA+negB) false-match rate at k does not exceed the k=1 false-match
rate at the default threshold 0.5. If no threshold below max(score) attains that target,
cap t_k at the largest attainable value strictly below the maximum score (reject every
negative but the top-scoring one) and record that the target was not reached.

python experiments/fix/fit_thresholds.py --raw data/raw --out experiments/fix/thresholds.json
"""
import argparse, json, os
import numpy as np
import pandas as pd

KS = [2, 4, 8, 16]
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
    ap.add_argument("--out", default="experiments/fix/thresholds.json")
    a = ap.parse_args()

    items = json.load(open(f"{a.raw}/items_all.json"))
    all_sirens = {it["siren"] for it in items}
    blocks = json.load(open(f"{a.raw}/blocks_e2e.json"))
    block_ids = {b["block"] for b in blocks}
    fit_sirens = sorted(all_sirens - block_ids)
    assert block_ids <= all_sirens
    print("fit entities:", len(fit_sirens), "excluded block ids:", len(block_ids), "total:", len(all_sirens))

    result = {"_meta": {"fit_entities": len(fit_sirens), "excluded_block_entities": len(block_ids),
                         "total_entities": len(all_sirens), "note": "k=1 bin is the reference point; t_1 fixed at 0.5"}}
    for m in MODELS:
        path = f"{a.raw}/scores_{m}.jsonl"
        if not os.path.exists(path):
            print("MISSING", path, "- skipping model"); continue
        d = pd.read_json(path, lines=True, dtype={"siren": str})
        d = d[d.siren.isin(fit_sirens)]
        neg1 = d[(d.kind == "single") & (d.x == 0) & (d.q.isin(NEG))].p_yes.values
        target = float((neg1 > 0.5).mean())
        bins = {"1": {"t": 0.5, "capped": False, "achieved_fmr": target, "note": "reference bin"}}
        for k in KS:
            negk = d[(d.kind == "set") & (d.x == k) & (d.q.isin(NEG))].p_yes.values
            assert len(negk) > 0, (m, k)
            t, info = fit_threshold(negk, target)
            bins[str(k)] = {"t": t, "n_neg": int(len(negk)), **info}
        result[m] = {"target_fmr_k1": target, "n_neg_k1": int(len(neg1)), "bins": bins}
        print(m, "target", round(target, 4), {k: round(v["t"], 4) for k, v in bins.items()})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
