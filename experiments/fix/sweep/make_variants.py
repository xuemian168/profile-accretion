"""Build per-model variants.json for the operating-curve sweep: 5 global profile_t<T>, 5 global
anchor_t<T>, 5 profile_cal_tau<tau> (bins from thresholds_tau.json), 1 profile_audit_soft.

python experiments/fix/sweep/make_variants.py --thresholds experiments/fix/sweep/thresholds_tau.json \
    --out-dir experiments/fix/sweep
"""
import argparse, json, os

GLOBAL_TS = [0.5, 0.8, 0.95, 0.99, 0.999]
TAUS = [0.05, 0.10, 0.15, 0.20, 0.30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    th = json.load(open(a.thresholds))
    models = [k for k in th if k != "_meta"]
    os.makedirs(a.out_dir, exist_ok=True)
    for m in models:
        variants = []
        for t in GLOBAL_TS:
            variants.append({"name": f"profile_t{t}", "use_profile": True, "threshold": t})
        for t in GLOBAL_TS:
            variants.append({"name": f"anchor_t{t}", "use_profile": False, "threshold": t})
        for tau in TAUS:
            bins = {k: v["t"] for k, v in th[m][str(tau)]["bins"].items()}
            variants.append({"name": f"profile_cal_tau{tau}", "use_profile": True, "cal_bins": bins})
        variants.append({"name": "profile_audit_soft", "use_profile": True, "threshold": 0.5,
                          "audit_soft": {"milestones": [4, 8, 16], "founder_t": 0.05, "loo_t": 0.5}})
        path = f"{a.out_dir}/variants_{m}.json"
        json.dump(variants, open(path, "w"), indent=2)
        print(m, "->", path, len(variants), "variants")


if __name__ == "__main__":
    main()
