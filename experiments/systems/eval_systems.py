"""Parse raw outputs from score_systems.py and compute false-match-rate /
recall vs k per strategy per model, with entity-level bootstrap CIs and
paired McNemar (k=1 vs k=16) tests. Writes data/derived/systems/summary.csv.

python eval_systems.py --raw_dir data/raw/systems --out data/derived/systems/summary.csv
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N_BOOT = 2000


def parse_batcher(completion, k):
    """Return list of k booleans (True=YES) or None if parse failed for that index."""
    ans = {}
    for m in re.finditer(r"Q\s*(\d+)\s*[:\.\)]\s*(YES|NO|Yes|No|yes|no)", completion):
        i = int(m.group(1))
        ans[i] = m.group(2).upper() == "YES"
    if len(ans) < k:
        # fallback: take first k standalone YES/NO tokens in order
        toks = re.findall(r"\b(YES|NO)\b", completion.upper())
        if len(toks) >= k:
            return [t == "YES" for t in toks[:k]], False
        return None, True  # parse failure
    return [ans.get(i) for i in range(1, k + 1)], any(v is None for v in ans.values())


def parse_llmcer(completion, query_idx, n_profile, n_recs):
    """Return predicted bool (query in same cluster as majority of profile ids) or None on parse fail."""
    matches = re.findall(r"\[([0-9,\s]+)\]", completion)
    clusters = []
    for m in matches:
        try:
            ids = [int(x) for x in re.findall(r"\d+", m)]
            if ids:
                clusters.append(ids)
        except ValueError:
            continue
    if not clusters:
        return None
    covered = set(x for c in clusters for x in c)
    if query_idx not in covered:
        return None
    profile_ids = set(range(1, n_profile + 1))
    query_cluster = None
    for c in clusters:
        if query_idx in c:
            query_cluster = set(c)
            break
    overlap = len(query_cluster & profile_ids)
    return overlap > (n_profile / 2.0)


def bootstrap_ci(values, sirens, n_boot=N_BOOT):
    """Entity-level bootstrap (resample unique sirens with replacement)."""
    df = pd.DataFrame({"siren": sirens, "v": values}).dropna()
    if df.empty:
        return (np.nan, np.nan, np.nan)
    uniq = df["siren"].unique()
    point = df["v"].mean()
    boots = []
    for _ in range(n_boot):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        sub = df[df["siren"].isin(samp)]
        # weight by resample multiplicity approx via groupby merge count
        cnt = pd.Series(samp).value_counts()
        w = df["siren"].map(cnt).fillna(0).values
        boots.append(np.average(df["v"].values, weights=w) if w.sum() > 0 else np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi


def mcnemar_exact(b, c):
    """Exact two-sided sign test (McNemar) given discordant counts b (1->0) and c (0->1)."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    pair_rows = []  # for McNemar: one row per (model,strategy,qtype,siren) with k1,k16 predictions

    # ---- ComEM select (next-token) ----
    for path in glob.glob(os.path.join(a.raw_dir, "comem_select_*.jsonl")):
        tag = os.path.basename(path)[len("comem_select_"):-len(".jsonl")]
        recs = [json.loads(l) for l in open(path)]
        df = pd.DataFrame(recs)
        df["pred_A"] = (df["p_A"] > df["p_B"]) & (df["p_A"] > df["p_C"])
        for (qtype, k), g in df.groupby(["qtype", "k"]):
            rows.append(dict(model=tag, strategy="comem_select", qtype=qtype, k=k,
                              n=len(g), rate=g["pred_A"].mean(), parse_fail_rate=0.0))
        piv = df.pivot_table(index=["siren", "qtype"], columns="k", values="pred_A", aggfunc="first").reset_index()
        for _, r in piv.iterrows():
            if 1 in piv.columns and 16 in piv.columns:
                pair_rows.append(dict(model=tag, strategy="comem_select", qtype=r["qtype"], siren=r["siren"],
                                       k1=r.get(1), k16=r.get(16)))

    # ---- BatchER (generation, OR over sub-answers) ----
    for path in glob.glob(os.path.join(a.raw_dir, "batcher_*.jsonl")):
        tag = os.path.basename(path)[len("batcher_"):-len(".jsonl")]
        recs = [json.loads(l) for l in open(path)]
        parsed = []
        for r in recs:
            ans, bad = parse_batcher(r["completion"], r["k"])
            pred = any(ans) if ans is not None else None
            parsed.append(dict(siren=r["siren"], qtype=r["qtype"], k=r["k"], label=r["label"],
                                pred=pred, parse_fail=ans is None))
        df = pd.DataFrame(parsed)
        for (qtype, k), g in df.groupby(["qtype", "k"]):
            ok = g.dropna(subset=["pred"])
            rows.append(dict(model=tag, strategy="batcher", qtype=qtype, k=k,
                              n=len(ok), rate=ok["pred"].mean() if len(ok) else np.nan,
                              parse_fail_rate=g["parse_fail"].mean()))
        piv = df.pivot_table(index=["siren", "qtype"], columns="k", values="pred", aggfunc="first").reset_index()
        for _, r in piv.iterrows():
            pair_rows.append(dict(model=tag, strategy="batcher", qtype=r["qtype"], siren=r["siren"],
                                   k1=r.get(1), k16=r.get(16)))

    # ---- LLM-CER cluster (generation, parse partition) ----
    for path in glob.glob(os.path.join(a.raw_dir, "llmcer_cluster_*.jsonl")):
        tag = os.path.basename(path)[len("llmcer_cluster_"):-len(".jsonl")]
        recs = [json.loads(l) for l in open(path)]
        parsed = []
        for r in recs:
            pred = parse_llmcer(r["completion"], r["query_idx"], r["n_profile"], r["n_recs"])
            parsed.append(dict(siren=r["siren"], qtype=r["qtype"], k=r["k"], label=r["label"],
                                pred=pred, parse_fail=pred is None))
        df = pd.DataFrame(parsed)
        for (qtype, k), g in df.groupby(["qtype", "k"]):
            ok = g.dropna(subset=["pred"])
            rows.append(dict(model=tag, strategy="llmcer_cluster", qtype=qtype, k=k,
                              n=len(ok), rate=ok["pred"].mean() if len(ok) else np.nan,
                              parse_fail_rate=g["parse_fail"].mean()))
        piv = df.pivot_table(index=["siren", "qtype"], columns="k", values="pred", aggfunc="first").reset_index()
        for _, r in piv.iterrows():
            pair_rows.append(dict(model=tag, strategy="llmcer_cluster", qtype=r["qtype"], siren=r["siren"],
                                   k1=r.get(1), k16=r.get(16)))

    summary = pd.DataFrame(rows)
    # attach bootstrap CI + McNemar per (model, strategy, qtype)
    extra = []
    pairs_df = pd.DataFrame(pair_rows)
    for (model, strategy, qtype), g in summary.groupby(["model", "strategy", "qtype"]):
        pr = pairs_df[(pairs_df.model == model) & (pairs_df.strategy == strategy) & (pairs_df.qtype == qtype)]
        pr = pr.dropna(subset=["k1", "k16"])
        b = int(((pr.k1 == True) & (pr.k16 == False)).sum())
        c = int(((pr.k1 == False) & (pr.k16 == True)).sum())
        pval = mcnemar_exact(b, c) if len(pr) else np.nan
        extra.append(dict(model=model, strategy=strategy, qtype=qtype,
                           mcnemar_k1_vs_k16_p=pval, n_paired=len(pr), discordant_0to1=c, discordant_1to0=b))
    extra_df = pd.DataFrame(extra)
    summary = summary.merge(extra_df, on=["model", "strategy", "qtype"], how="left")

    cis = []
    for path_glob, strat_name in [("comem_select_*.jsonl", "comem_select"),
                                   ("batcher_*.jsonl", "batcher"),
                                   ("llmcer_cluster_*.jsonl", "llmcer_cluster")]:
        pass  # CI computed inline above is point estimate only; add bootstrap below
    summary["ci_lo"] = np.nan
    summary["ci_hi"] = np.nan
    for idx, row in summary.iterrows():
        pr = pairs_df[(pairs_df.model == row["model"]) & (pairs_df.strategy == row["strategy"]) & (pairs_df.qtype == row["qtype"])]
        col = "k1" if row["k"] == 1 else ("k16" if row["k"] == 16 else None)
        if col is None:
            continue
        vals = pr[col].astype(float)
        point, lo, hi = bootstrap_ci(vals.values, pr["siren"].values)
        summary.loc[idx, "ci_lo"] = lo
        summary.loc[idx, "ci_hi"] = hi

    summary = summary.sort_values(["model", "strategy", "qtype", "k"])
    summary.to_csv(a.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
