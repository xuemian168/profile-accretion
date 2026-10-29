"""Precision-matched operating point for whole-profile online linking.

For each model, the candidate operating points are the ten whole-profile sweep variants
(global thresholds 0.5-0.999 and size-calibrated thresholds tau 0.05-0.3 fit on the 186
held-out entities).  In 5-fold cross-validation over the 80 blocks (20 repetitions) we pick,
on the training blocks, the highest-recall profile point whose pairwise precision is at least
that of the founding-member rule (anchor, t=0.5) on the same blocks, and evaluate it on the
held-out blocks against the founding-member rule.  95% (and 90%, for a one-sided 95% non-inferiority bound) intervals: block bootstrap (2,000) of
the out-of-fold counts from the first repetition.
Also writes join precision by cluster size at join time.
Outputs: data/derived/fix/sweep/cv_operating_point.csv, join_precision_by_size.csv
"""
import glob, os, json
from collections import Counter
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RAW, OUT = f"{ROOT}/data/raw/fix/sweep", f"{ROOT}/data/derived/fix/sweep"
MODELS = ['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct', 'Qwen2.5-32B-Instruct', 'Qwen3-32B',
          'OLMo-2-0325-32B-Instruct', 'Mistral-Small-24B-Instruct-2501']
GLB = [f'profile_t{t}' for t in ['0.5', '0.8', '0.95', '0.99', '0.999']]
CAL = [f'profile_cal_tau{t}' for t in ['0.05', '0.1', '0.15', '0.2', '0.3']]
OPS = GLB + CAL
c2 = lambda n: n * (n - 1) // 2

def pair_counts(g):
    both = Counter(zip(g.cluster, g.siren)); tp = sum(c2(n) for n in both.values())
    pp = sum(c2(n) for n in Counter(g.cluster).values()); ap = sum(c2(n) for n in Counter(g.siren).values())
    return tp, pp - tp, ap - tp

def prf(t):
    tp, fp, fn = t
    return (tp / (tp + fp) if tp + fp else 0.0), tp / (tp + fn)

def size_bin(s):
    return '1' if s == 1 else ('2-3' if s < 4 else ('4-7' if s < 8 else '8+'))

CNT, JP, blocks = {}, [], None
for m in MODELS:
    d = pd.read_json(f"{RAW}/sweep_{m}.jsonl", lines=True, dtype={'siren': str, 'block': str})
    d = d[d.variant != 'profile_audit_soft']
    for v, g in d.groupby('variant'):
        bc = {b: pair_counts(x) for b, x in g.groupby('block')}
        blocks = blocks or sorted(bc)
        CNT[(m, v)] = np.array([bc[b] for b in blocks], dtype=float)
        founder, rows = {}, []
        for r in g.sort_values(['block', 'i']).itertuples():
            key = (r.block, r.cluster)
            if not r.joined: founder[key] = r.siren
            else: rows.append((size_bin(r.size_before), r.siren == founder.get(key)))
        x = pd.DataFrame(rows, columns=['bin', 'ok'])
        for b_, gg in x.groupby('bin'):
            JP.append(dict(model=m, variant=v, size_bin=b_, n_joins=len(gg), join_precision=gg.ok.mean()))
pd.DataFrame(JP).to_csv(f"{OUT}/join_precision_by_size.csv", index=False)
NB = len(blocks)

def cv_once(m, rng):
    folds = np.array_split(rng.permutation(NB), 5); A = CNT[(m, 'anchor_t0.5')]
    sel, anc, chosen = np.zeros_like(A), np.zeros_like(A), []
    for te in folds:
        tr = np.setdiff1d(np.arange(NB), te); pa, _ = prf(A[tr].sum(0))
        cands = [(prf(CNT[(m, v)][tr].sum(0)), v) for v in OPS]
        ok = [(r, p, v) for (p, r), v in cands if p >= pa]
        v = max(ok)[2] if ok else max(cands, key=lambda c: c[0][0])[1]
        chosen.append(v); sel[te] = CNT[(m, v)][te]; anc[te] = A[te]
    return sel, anc, chosen

rows = []
for m in MODELS:
    rng = np.random.default_rng(7); reps = []
    for rep in range(20):
        sel, anc, ch = cv_once(m, rng)
        (ps, rs), (pa, ra) = prf(sel.sum(0)), prf(anc.sum(0))
        reps.append((ps - pa, rs / ra, rs - ra, ps, rs)); chosen = chosen if rep else ch
        if rep == 0:
            b = []
            for _ in range(2000):
                i = rng.integers(0, NB, NB); (p1, r1), (p0, r0) = prf(sel[i].sum(0)), prf(anc[i].sum(0)); b.append((p1 - p0, r1 - r0))
            b = np.array(b); ci = np.percentile(b, [2.5, 97.5], axis=0); ci90 = np.percentile(b, [5, 95], axis=0)
    X = np.array(reps); pa, ra = prf(CNT[(m, 'anchor_t0.5')].sum(0))
    rows.append(dict(model=m, anchor_precision=pa, anchor_recall=ra, profile_precision=X[:, 3].mean(), profile_recall=X[:, 4].mean(),
                     d_precision=X[:, 0].mean(), d_precision_lo=ci[0, 0], d_precision_hi=ci[1, 0], d_precision_lo90=ci90[0, 0], d_precision_hi90=ci90[1, 0],
                     recall_ratio=X[:, 1].mean(), d_recall=X[:, 2].mean(), d_recall_lo=ci[0, 1], d_recall_hi=ci[1, 1],
                     chosen_rep1=';'.join(chosen)))
pd.DataFrame(rows).to_csv(f"{OUT}/cv_operating_point.csv", index=False)
print(pd.DataFrame(rows).drop(columns='chosen_rep1').round(3).to_string())
