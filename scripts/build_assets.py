"""Compute every number, table and figure in the paper from data/raw.

python scripts/build_assets.py      (run from the repository root)
Outputs: generated/numbers.tex, generated/tab_*.tex, data/derived/*.csv, figures/fig*.pdf/png
"""
import json, os, re
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW, DER, GEN, FIG = "data/raw", "data/derived", "generated", "figures"
for d in (DER, GEN, FIG): os.makedirs(d, exist_ok=True)
KS = [1, 2, 4, 8, 16]
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-32B"]
EXTRA = ["OLMo-2-0325-32B-Instruct", "Mistral-Small-24B-Instruct-2501"]               # non-Qwen replication
MODELS += [m for m in EXTRA if all(os.path.exists(f"data/raw/{p}_{m}.jsonl") for p in ("scores", "mitig", "e2e"))]
SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen2.5-14B-Instruct": "Qwen2.5-14B", "Qwen2.5-32B-Instruct": "Qwen2.5-32B", "Qwen3-32B": "Qwen3-32B", "OLMo-2-0325-32B-Instruct": "OLMo-2-32B", "Mistral-Small-24B-Instruct-2501": "Mistral-24B"}
KEY = {"Qwen2.5-7B-Instruct": "Sev", "Qwen2.5-14B-Instruct": "Fourt", "Qwen2.5-32B-Instruct": "Thirt", "Qwen3-32B": "Qthree", "OLMo-2-0325-32B-Instruct": "Olmo", "Mistral-Small-24B-Instruct-2501": "Mis"}
COL = {"Qwen2.5-7B-Instruct": "#56B4E9", "Qwen2.5-14B-Instruct": "#009E73", "Qwen2.5-32B-Instruct": "#0B3C8C", "Qwen3-32B": "#D55E00", "OLMo-2-0325-32B-Instruct": "#CC79A7", "Mistral-Small-24B-Instruct-2501": "#999933"}
MKM = {"Qwen2.5-7B-Instruct": "o", "Qwen2.5-14B-Instruct": "s", "Qwen2.5-32B-Instruct": "^", "Qwen3-32B": "D", "OLMo-2-0325-32B-Instruct": "v", "Mistral-Small-24B-Instruct-2501": "P"}
LSM = {"Qwen2.5-7B-Instruct": "-", "Qwen2.5-14B-Instruct": "--", "Qwen2.5-32B-Instruct": "-.", "Qwen3-32B": (0, (1, 1)), "OLMo-2-0325-32B-Instruct": (0, (5, 1, 1, 1, 1, 1)), "Mistral-Small-24B-Instruct-2501": (0, (3, 1))}
NEG = ["negA", "negB"]

macros = {}
def mac(name, val, fmt="{:.1f}"):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    v = val if isinstance(val, str) else fmt.format(val)
    if not isinstance(val, str) and v[:1] in "+-": v = ("\\textminus{}" if v[0] == "-" else "+") + v[1:]
    macros[name] = v
pct = lambda x: 100 * x
def sci(p):  # LaTeX math-mode p-value
    if p >= 1e-3: return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e"); return f"{m}\\times10^{{{int(e)}}}"

items = {it["siren"]: it for it in json.load(open(f"{RAW}/items_all.json"))}
stats = json.load(open(f"{RAW}/foppa_stats.json"))
nm = lambda t: t.split(" | ")[0]
exact = {(s, q): nm(it["queries"][q]["text"]) in {nm(m) for m in it["members"]} for s, it in items.items() for q in NEG}

def tables(d):
    S = d[d.kind == "single"].pivot_table(index=["siren", "q"], columns="x", values="p_yes")
    T = d[d.kind == "set"].pivot_table(index=["siren", "q"], columns="x", values="p_yes"); T[1] = S[0]
    R = d[d.kind == "set_rev"].set_index(["siren", "q"]).p_yes
    return S, T, R

def split(df, qs):
    return df[df.index.get_level_values("q").isin(qs)]

def mcnemar(a, b):
    up = int(((~a) & b).sum()); dn = int((a & ~b).sum())
    return up, dn, (binomtest(up, up + dn).pvalue if up + dn else 1.0)

# ---------------------------------------------------------------- controlled accretion
rows, conf_rows, matched_rows, order_rows = [], [], [], []
MCP = {}
SC = {}
for m in MODELS:
    d = pd.read_json(f"{RAW}/scores_{m}.jsonl", lines=True, dtype={"siren": str}); assert len(d) == 16758
    S, T, R = tables(d); SC[m] = (S, T)
    Sn, Tn = split(S, NEG), split(T, NEG); Sp, Tp = split(S, ["pos"]), split(T, ["pos"])
    ex = np.array([exact[i] for i in Tn.index])
    for k in KS:
        yn = Tn[k] > .5; orr = (Sn.iloc[:, :k] > .5).any(axis=1)
        auc = roc_auc_score(np.r_[np.ones(len(Tp)), np.zeros(len(Tn))], np.r_[Tp[k].values, Tn[k].values])
        rows.append(dict(model=m, k=k, fmr=yn.mean(), fmr_or=orr.mean(), agree_or=(yn == orr).mean(),
                         fmr_distinct=yn[~ex].mean(), fmr_exactname=yn[ex].mean(),
                         fmr_A=(split(T, ["negA"])[k] > .5).mean(), fmr_B=(split(T, ["negB"])[k] > .5).mean(),
                         recall=(Tp[k] > .5).mean(), auc=auc, n_neg=len(Tn), n_pos=len(Tp)))
    up, dn, p = mcnemar(Tn[1] > .5, Tn[16] > .5)
    mac(f"mcUp{KEY[m]}", str(up)); mac(f"mcDown{KEY[m]}", str(dn)); mac(f"mcP{KEY[m]}", sci(p))
    MCP[m] = p
    conf_rows.append(dict(model=m, k1=(Tn[1] > .99).mean(), best_single16=(Sn.max(axis=1) > .99).mean(), profile16=(Tn[16] > .99).mean()))
    target = (Tn[1] > .5).mean()
    for k in KS:
        t = .5 if k == 1 else float(np.quantile(Tn[k].values, 1 - target))
        matched_rows.append(dict(model=m, k=k, threshold=t, fmr=(Tn[k] > t).mean(), recall=(Tp[k] > t).mean()))
    order_rows.append(dict(model=m, mean_abs_dp=float(np.abs(T[16] - R.reindex(T.index)).mean()),
                           fmr16=(Tn[16] > .5).mean(), fmr16_rev=(R.reindex(Tn.index) > .5).mean()))
C = pd.DataFrame(rows); CF = pd.DataFrame(conf_rows); MF = pd.DataFrame(matched_rows); OR_ = pd.DataFrame(order_rows)
C.to_csv(f"{DER}/accretion_curves.csv", index=False); CF.to_csv(f"{DER}/confident_false_matches.csv", index=False)
MF.to_csv(f"{DER}/matched_fmr_recall.csv", index=False); OR_.to_csv(f"{DER}/order_control.csv", index=False)

for m in MODELS:
    c = C[C.model == m].set_index("k"); K = KEY[m]
    for k, w in [(1, "one"), (2, "two"), (16, "sixteen")]:
        mac(f"fmr{w}{K}", pct(c.loc[k, "fmr"])); mac(f"rec{w}{K}", pct(c.loc[k, "recall"])); mac(f"auc{w}{K}", c.loc[k, "auc"], "{:.2f}")
        mac(f"orn{w}{K}", pct(c.loc[k, "fmr_or"])); mac(f"dis{w}{K}", pct(c.loc[k, "fmr_distinct"]))
        mac(f"agr{w}{K}", pct(c.loc[k, "agree_or"]), "{:.0f}")
    mac(f"ratio{K}", c.loc[16, "fmr"] / c.loc[1, "fmr"], "{:.1f}"); mac(f"recratio{K}", c.loc[16, "recall"] / c.loc[1, "recall"], "{:.1f}")
    cf = CF.set_index("model").loc[m]
    mac(f"cfone{K}", pct(cf.k1)); mac(f"cfbest{K}", pct(cf.best_single16)); mac(f"cfprof{K}", pct(cf.profile16))
    mf = MF[MF.model == m].set_index("k")
    mac(f"mrone{K}", pct(mf.loc[1, "recall"])); mac(f"mrsixteen{K}", pct(mf.loc[16, "recall"])); mac(f"mfone{K}", pct(mf.loc[1, "fmr"]))
    mac(f"ordp{K}", OR_.set_index("model").loc[m, "mean_abs_dp"], "{:.2f}")
mac("ratioMin", C[C.k == 16].set_index("model").fmr.div(C[C.k == 1].set_index("model").fmr).min(), "{:.1f}")
mac("ratioMax", C[C.k == 16].set_index("model").fmr.div(C[C.k == 1].set_index("model").fmr).max(), "{:.1f}")
mac("recratioMin", C[C.k == 16].set_index("model").recall.div(C[C.k == 1].set_index("model").recall).min(), "{:.1f}")
mac("recratioMax", C[C.k == 16].set_index("model").recall.div(C[C.k == 1].set_index("model").recall).max(), "{:.1f}")
mac("aucGainMin", (C[C.k == 16].set_index("model").auc - C[C.k == 1].set_index("model").auc).min(), "{:.2f}")
mac("aucGainMax", (C[C.k == 16].set_index("model").auc - C[C.k == 1].set_index("model").auc).max(), "{:.2f}")
mac("ordMax", OR_.mean_abs_dp.max(), "{:.2f}")
mac("mcPmax", sci(max(MCP.values())))
q25 = [m for m in MODELS if m.startswith("Qwen2.5")]
dmr = [pct(MF[(MF.model == m) & (MF.k == 16)].recall.iloc[0] - MF[(MF.model == m) & (MF.k == 1)].recall.iloc[0]) for m in q25]
mac("mrDeltaMax", max(dmr), "{:+.1f}"); mac("mrDeltaMin", min(dmr), "{:+.1f}")
exc = (C.fmr - C.fmr_or) * 100
for m in MODELS:
    e = exc[(C.model == m) & (C.k > 1)]; mac(f"excMax{KEY[m]}", e.max()); mac(f"excMin{KEY[m]}", e.min())
mac("massMin", min(pd.read_json(f"{RAW}/scores_{m}.jsonl", lines=True).mass.min() for m in MODELS), "{:.3f}")
mac("nExactA", str(sum(exact[(s, "negA")] for s in items))); mac("nExactB", str(sum(exact[(s, "negB")] for s in items)))
mac("nNeg", str(int(C.n_neg.iloc[0]))); mac("nPos", str(int(C.n_pos.iloc[0])))

# ---------------------------------------------------------------- mitigation and contamination
mit_rows, con_rows = [], []
for m in MODELS:
    mi = pd.read_json(f"{RAW}/mitig_{m}.jsonl", lines=True, dtype={"siren": str}); assert len(mi) == 6690
    S, T = SC[m]
    def rates(kind, x=None):
        sub = mi[(mi.kind == kind) & ((mi.x == x) if x is not None else True)]
        return (sub[sub.q.isin(NEG)].p_yes > .5).mean(), (sub[sub.q == "pos"].p_yes > .5).mean()
    variants = {"founding member only": ((split(T, NEG)[1] > .5).mean(), (split(T, ["pos"])[1] > .5).mean()),
                "whole profile (k=16)": ((split(T, NEG)[16] > .5).mean(), (split(T, ["pos"])[16] > .5).mean()),
                "skeptical prompt (k=16)": rates("skeptical", 16),
                "last 4 members": rates("bounded_last4"),
                "founding + last 3": rates("anchor_plus3")}
    for v, (f_, r_) in variants.items(): mit_rows.append(dict(model=m, variant=v, fmr=f_, recall=r_))
    for k in [4, 16]:
        cl = mi[(mi.kind == "clean") & (mi.x == k)].set_index(["siren", "q"]).p_yes
        co = mi[(mi.kind == "contaminated") & (mi.x == k)].set_index(["siren", "q"]).p_yes.reindex(cl.index)
        up, dn, p = mcnemar(cl > .5, co > .5)
        con_rows.append(dict(model=m, k=k, n=len(cl), clean=(cl > .5).mean(), contaminated=(co > .5).mean(), up=up, down=dn, p=p))
MI = pd.DataFrame(mit_rows); CO = pd.DataFrame(con_rows)
MI.to_csv(f"{DER}/mitigation.csv", index=False); CO.to_csv(f"{DER}/contamination.csv", index=False)
for m in MODELS:
    K = KEY[m]; mm = MI[MI.model == m].set_index("variant")
    mac(f"skep{K}", pct(mm.loc["skeptical prompt (k=16)", "fmr"])); mac(f"lastf{K}", pct(mm.loc["last 4 members", "fmr"]))
    mac(f"ancth{K}", pct(mm.loc["founding + last 3", "fmr"]))
    for k, w in [(4, "four"), (16, "sixteen")]:
        c = CO[(CO.model == m) & (CO.k == k)].iloc[0]
        mac(f"cl{w}{K}", pct(c.clean)); mac(f"co{w}{K}", pct(c.contaminated))
        mac(f"coUp{w}{K}", str(c.up)); mac(f"coDown{w}{K}", str(c.down))
mac("nSecond", str(int(CO.n.iloc[0])))
mac("coPmax", sci(CO.p.max()))

# ---------------------------------------------------------------- end-to-end online linking
def pair_counts(g):
    tp = fp = fn = 0
    for _, b in g.groupby("block"):
        cl = b.cluster.values; tr = b.siren.values; iu = np.triu_indices(len(b), 1)
        sc = (cl[:, None] == cl[None, :])[iu]; st = (tr[:, None] == tr[None, :])[iu]
        tp += int((sc & st).sum()); fp += int((sc & ~st).sum()); fn += int((~sc & st).sum())
    return tp, fp, fn
e2e_rows = []
for m in MODELS:
    d = pd.read_json(f"{RAW}/e2e_{m}.jsonl", lines=True, dtype={"siren": str, "block": str})
    for rep, g in d.groupby("rep"):
        tp, fp, fn = pair_counts(g); p, r = tp / (tp + fp), tp / (tp + fn)
        cl = g.groupby(["block", "cluster"]).siren.agg(list)
        e2e_rows.append(dict(model=m, rep=rep, precision=p, recall=r, f1=2 * p * r / (p + r), clusters=len(cl),
                             mixed=int(cl.map(lambda s: len(set(s)) > 1).sum())))
E = pd.DataFrame(e2e_rows); E.to_csv(f"{DER}/e2e.csv", index=False)
for m in MODELS:
    K = KEY[m]; e = E[E.model == m].set_index("rep")
    for rep, w in [("anchor", "anc"), ("profile", "prof"), ("pairmax", "pmax")]:
        mac(f"eP{w}{K}", e.loc[rep, "precision"], "{:.2f}"); mac(f"eR{w}{K}", e.loc[rep, "recall"], "{:.2f}"); mac(f"eF{w}{K}", e.loc[rep, "f1"], "{:.2f}")
blocks = json.load(open(f"{RAW}/blocks_e2e.json"))
rr = E.pivot_table(index="model", columns="rep", values="recall"); rr = rr.profile / rr.anchor
mac("eRatioMin", rr.min(), "{:.1f}"); mac("eRatioMax", rr.max(), "{:.1f}")
mac("nBlocks", str(len(blocks))); mac("nBlockRecords", f"{sum(len(b['records']) for b in blocks):,}")
mac("nBlockSirens", f"{len({r['siren'] for b in blocks for r in b['records']}):,}")

# ---------------------------------------------------------------- Claude Haiku pilot (generated answers)
import glob
H = pd.concat([pd.read_json(f, lines=True, dtype={"siren": str}) for f in sorted(glob.glob(f"{RAW}/haiku_pilot_*.jsonl"))])
H = H[H.err.isna() & H.ans.notna()]; full = H.groupby("siren").size(); full = full[full == 45].index; H = H[H.siren.isin(full)]
HS = H[H.kind == "single"].pivot_table(index=["siren", "q"], columns="x", values="ans")
HT = H[H.kind == "set"].pivot_table(index=["siren", "q"], columns="x", values="ans"); HT[1] = HS[0]
hn, hsn, hp = split(HT, NEG), split(HS, NEG), split(HT, ["pos"])
mac("hN", str(len(full))); mac("hfone", pct(hn[1].mean())); mac("hfsixteen", pct(hn[16].mean()))
mac("hor", pct(hsn.iloc[:, :16].max(axis=1).mean())); mac("hrone", pct(hp[1].mean())); mac("hrsixteen", pct(hp[16].mean()))
mac("hagr", pct(np.mean([(hn[k] == hsn.iloc[:, :k].max(axis=1)).mean() for k in KS[1:]])), "{:.0f}")
up, dn, p = mcnemar(hn[1] == 1, hn[16] == 1); mac("hp", sci(p)); mac("hUp", str(up)); mac("hDown", str(dn))

# ---------------------------------------------------------------- robustness and uncertainty (reviewer-driven)
from scipy.stats import spearmanr
pop = json.load(open(f"{RAW}/foppa_population.json"))
ENT = sorted(items); B = 2000; rng_b = np.random.default_rng(2027)
boot_idx = rng_b.integers(0, len(ENT), size=(B, len(ENT)))
def ci(a): return np.percentile(a, [2.5, 97.5])
def sign_p(d):
    d = np.asarray(d); up, dn = int((d > 0).sum()), int((d < 0).sum())
    return binomtest(up, up + dn).pvalue if up + dn else 1.0
rob_rows, single_rows, cv_rows = [], [], []
tests_p = []
for m in MODELS:
    S, T = SC[m]; K = KEY[m]
    neg = {k: np.stack([T.loc[[(s, q) for s in ENT], k].values for q in NEG], 1) for k in KS}      # entity x 2
    pos = {k: T.loc[[(s, "pos") for s in ENT], k].values for k in KS}
    f1_, f16 = (neg[1] > .5).mean(1), (neg[16] > .5).mean(1); r1, r16 = pos[1] > .5, pos[16] > .5
    bt = {"fmr1": [], "fmr16": [], "dfmr": [], "rec1": [], "rec16": [], "dauc": []}
    for ix in boot_idx:
        a1, a16 = f1_[ix].mean(), f16[ix].mean(); bt["fmr1"].append(a1); bt["fmr16"].append(a16); bt["dfmr"].append(a16 - a1)
        bt["rec1"].append(r1[ix].mean()); bt["rec16"].append(r16[ix].mean())
        y = np.r_[np.ones(len(ix)), np.zeros(2 * len(ix))]
        au = [roc_auc_score(y, np.r_[pos[k][ix], neg[k][ix].ravel()]) for k in (1, 16)]; bt["dauc"].append(au[1] - au[0])
    p_ent = sign_p(f16 - f1_); tests_p.append(p_ent)
    row = dict(model=m, p_entity=p_ent, n_ent_up=int((f16 > f1_).sum()), n_ent_down=int((f16 < f1_).sum()))
    for k_, v in bt.items(): lo, hi = ci(v); row[k_ + "_lo"], row[k_ + "_hi"] = lo, hi
    rob_rows.append(row)
    mac(f"dfmr{K}", pct(f16.mean() - f1_.mean())); mac(f"dfmrLo{K}", pct(row["dfmr_lo"])); mac(f"dfmrHi{K}", pct(row["dfmr_hi"]))
    mac(f"pEnt{K}", sci(p_ent)); mac(f"entUp{K}", str(row["n_ent_up"])); mac(f"entDown{K}", str(row["n_ent_down"]))
    # single non-founding members judged alone
    Sn, Sp = split(S, NEG), split(S, ["pos"])
    fm = [(Sn[j] > .5).mean() for j in range(16)]; rc = [(Sp[j] > .5).mean() for j in range(16)]
    fA = [(split(S, ["negA"])[j] > .5).mean() for j in range(16)]; fB = [(split(S, ["negB"])[j] > .5).mean() for j in range(16)]
    single_rows.append(dict(model=m, fmr_founding=fm[0], fmr_other_mean=np.mean(fm[1:]), fmr_other_min=min(fm[1:]), fmr_other_max=max(fm[1:]),
                            rec_founding=rc[0], rec_other_mean=np.mean(rc[1:]), fmrA_founding=fA[0], fmrA_other=np.mean(fA[1:]),
                            fmrB_founding=fB[0], fmrB_other=np.mean(fB[1:])))
    mac(f"sgOther{K}", pct(np.mean(fm[1:]))); mac(f"sgOtherMin{K}", pct(min(fm[1:]))); mac(f"sgOtherMax{K}", pct(max(fm[1:])))
    mac(f"sgFound{K}", pct(fm[0])); mac(f"sgRecOther{K}", pct(np.mean(rc[1:])))
    # cross-validated matched recall: 5 folds over entities x 20 repeats
    rng_cv = np.random.default_rng(99); res = []
    for rep in range(20):
        perm = rng_cv.permutation(len(ENT)); folds = np.array_split(perm, 5)
        for f in folds:
            tr = np.setdiff1d(np.arange(len(ENT)), f)
            target = (neg[1][tr] > .5).mean(); v = np.sort(neg[16][tr].ravel())
            cand = np.unique(v); fm_at = np.array([(v > t).mean() for t in cand])
            ok = cand[(fm_at <= target) & (cand < v.max())]          # rejecting every record does not count
            if len(ok) == 0: res.append(dict(attain=False)); continue
            t = ok.min()
            res.append(dict(attain=True, fmr16=(neg[16][f] > t).mean(), rec16=(pos[16][f] > t).mean(),
                            fmr1=(neg[1][f] > .5).mean(), rec1=(pos[1][f] > .5).mean()))
    R_ = pd.DataFrame(res); att = R_.attain.mean()
    if att > 0:
        A_ = R_[R_.attain]; cv_rows.append(dict(model=m, attain=att, fmr1=A_.fmr1.mean(), fmr16=A_.fmr16.mean(), rec1=A_.rec1.mean(), rec16=A_.rec16.mean()))
    else: cv_rows.append(dict(model=m, attain=0.0, fmr1=(neg[1] > .5).mean(), fmr16=np.nan, rec1=(pos[1] > .5).mean(), rec16=np.nan))
    mac(f"cvAtt{K}", pct(att), "{:.0f}")
    if att > 0: mac(f"cvRone{K}", pct(cv_rows[-1]["rec1"])); mac(f"cvRsix{K}", pct(cv_rows[-1]["rec16"])); mac(f"cvFsix{K}", pct(cv_rows[-1]["fmr16"])); mac(f"cvFone{K}", pct(cv_rows[-1]["fmr1"]))
    # effect vs entity size
    nag = np.array([pop["agents_per_entity"][s] for s in ENT]); rho, prho = spearmanr(nag, f16 - f1_)
    rob_rows[-1].update(rho_size=rho, p_rho=prho); mac(f"rho{K}", rho, "{:+.2f}")
RB = pd.DataFrame(rob_rows); SG = pd.DataFrame(single_rows); CV = pd.DataFrame(cv_rows)
RB.to_csv(f"{DER}/bootstrap_entity.csv", index=False); SG.to_csv(f"{DER}/single_member_baseline.csv", index=False); CV.to_csv(f"{DER}/matched_recall_cv.csv", index=False)
mac("rhoMin", RB.rho_size.min(), "{:+.2f}"); mac("rhoMax", RB.rho_size.max(), "{:+.2f}"); mac("prhoMin", RB.p_rho.min(), "{:.2f}")
q25cv = CV[~CV.model.str.startswith("Qwen3") & (CV.attain > 0)]   # models without a clear matched-recall gain
mac("cvDeltaMax", pct((q25cv.rec16 - q25cv.rec1).max()), "{:+.1f}"); mac("cvDeltaMin", pct((q25cv.rec16 - q25cv.rec1).min()), "{:+.1f}")
mac("dfmrMin", pct((C[C.k == 16].set_index("model").fmr - C[C.k == 1].set_index("model").fmr).min()))
mac("drecMin", pct((C[C.k == 16].set_index("model").recall - C[C.k == 1].set_index("model").recall).min()))
mac("drecMax", pct((C[C.k == 16].set_index("model").recall - C[C.k == 1].set_index("model").recall).max()))
mac("dfmrMax", pct((C[C.k == 16].set_index("model").fmr - C[C.k == 1].set_index("model").fmr).max()))
g = (C[C.k == 16].set_index("model").auc - C[C.k == 1].set_index("model").auc) / (1 - C[C.k == 1].set_index("model").auc)
mac("aucRelMin", pct(g.min()), "{:.0f}"); mac("aucRelMax", pct(g.max()), "{:.0f}")
mac("sgOtherMinAll", pct(SG.fmr_other_mean.min())); mac("sgOtherMaxAll", pct(SG.fmr_other_mean.max()))
# contamination at entity level
for m in MODELS:
    mi = pd.read_json(f"{RAW}/mitig_{m}.jsonl", lines=True, dtype={"siren": str})
    for k in [4, 16]:
        cl = mi[(mi.kind == "clean") & (mi.x == k)].groupby("siren").p_yes.apply(lambda x: (x > .5).mean())
        co = mi[(mi.kind == "contaminated") & (mi.x == k)].groupby("siren").p_yes.apply(lambda x: (x > .5).mean()).reindex(cl.index)
        tests_p.append(sign_p(co - cl))
mac("nTests", str(len(tests_p))); mac("pEntMax", sci(max(tests_p))); mac("pEntBonf", sci(min(1, max(tests_p) * len(tests_p))))
# order reversal distribution over all queries
od = []
for m in MODELS:
    d = pd.read_json(f"{RAW}/scores_{m}.jsonl", lines=True, dtype={"siren": str}); S_, T_, R_r = tables(d)
    dp = np.abs(T_[16] - R_r.reindex(T_.index)).dropna(); od.append(dict(model=m, median=dp.median(), p90=dp.quantile(.9), n=len(dp),
        dfmr=pct((R_r.reindex(split(T_, NEG).index) > .5).mean() - (split(T_, NEG)[16] > .5).mean())))
OD = pd.DataFrame(od); OD.to_csv(f"{DER}/order_control_dist.csv", index=False)
mac("ordMedMax", OD["median"].max(), "{:.3f}"); mac("ordPninetyMax", OD.p90.max(), "{:.2f}")
mac("ordDfmrMin", OD.dfmr.min(), "{:+.1f}"); mac("ordDfmrMax", OD.dfmr.max(), "{:+.1f}")
# online linking: candidate (blocking) recall
br = []
for m in MODELS:
    d = pd.read_json(f"{RAW}/e2e_{m}.jsonl", lines=True, dtype={"siren": str, "block": str})
    for rep, g_ in d.groupby("rep"):
        need = hit = 0
        for _, b in g_.groupby("block"):
            b = b.sort_values("i"); seen = []
            for r in b.itertuples():
                true_cl = {c for s, c in seen if s == r.siren}
                if true_cl:
                    need += 1; hit += bool(true_cl & {int(c) for c in (r.cand_scores or {}).keys()})
                seen.append((r.siren, int(r.cluster)))
        br.append(dict(model=m, rep=rep, need=need, cand_recall=hit / need))
BR = pd.DataFrame(br); BR.to_csv(f"{DER}/e2e_candidate_recall.csv", index=False)
mac("candRecMin", pct(BR.cand_recall.min())); mac("candRecMax", pct(BR.cand_recall.max()))
# identical-name negatives: same location as the matching member?
loc = lambda t: t.split(" | ")[1] if " | " in t else ""
same_loc = [loc(it["queries"][q]["text"]) in {loc(mm) for mm, n_ in zip(it["members"], it["members"]) if nm(mm) == nm(it["queries"][q]["text"])}
            for s, it in items.items() for q in NEG if exact[(s, q)]]
mac("exSameLoc", pct(np.mean(same_loc)), "{:.0f}"); mac("nExact", str(len(same_loc)))
# population
mac("popSingle", pct(pop["share_siren_single_agent"]), "{:.0f}"); mac("popAgents", pct(pop["share_agents_in_ge17"]))
mac("popLots", pct(pop["share_lots_in_ge17"])); mac("entMedAgents", f"{pop['agents_ge17_quantiles']['0.5']:.0f}"); mac("entMaxAgents", f"{pop['agents_ge17_quantiles']['1']:.0f}")
# contamination gap
for m in MODELS:
    K = KEY[m]; c4 = CO[(CO.model == m) & (CO.k == 4)].iloc[0]; c16 = CO[(CO.model == m) & (CO.k == 16)].iloc[0]
    mac(f"cogapfour{K}", pct(c4.contaminated - c4.clean)); mac(f"cogapsixteen{K}", pct(c16.contaminated - c16.clean))

# distance of each mitigation from the model's own k=1..16 curve (linear interpolation, within the curve's FMR range)
dist = []
for m in MODELS:
    c = C[C.model == m].sort_values("fmr")
    for v in ["skeptical prompt (k=16)", "last 4 members", "founding + last 3"]:
        r = MI[(MI.model == m) & (MI.variant == v)].iloc[0]
        inside = c.fmr.min() <= r.fmr <= c.fmr.max()
        dist.append(dict(model=m, variant=v, fmr=r.fmr, recall=r.recall, inside=inside,
                         d=(r.recall - np.interp(r.fmr, c.fmr, c.recall)) if inside else np.nan))
DS = pd.DataFrame(dist); DS.to_csv(f"{DER}/mitigation_curve_distance.csv", index=False)
b = DS[DS.variant != "skeptical prompt (k=16)"].dropna(); sk = DS[DS.variant == "skeptical prompt (k=16)"].dropna()
mac("mitBoundMax", pct(b.d.abs().max()), "{:.0f}"); mac("mitSkepMax", pct(sk.d.max()), "{:.0f}")
mac("mitSkepMaxModel", SHORT[sk.loc[sk.d.idxmax(), "model"]])
# numerical saturation of P(YES) at k=16
for m in MODELS:
    S, T = SC[m]; K = KEY[m]; tn = split(T, NEG)[16].values; tp_ = split(T, ["pos"])[16].values
    mac(f"satNeg{K}", pct((tn >= tn.max()).mean())); mac(f"satPos{K}", pct((tp_ >= tn.max()).mean()))
# ---------------------------------------------------------------- robustness checks from the extension tracks
CHK = []
def rng_pp(v): v = 100 * np.asarray(v, float); return f"{v.min():+.1f} to {v.max():+.1f}".replace("+", "$+$").replace("-", "$-$")
def sig_up(lo): return int((np.asarray(lo) > 0).sum())
AD = pd.read_csv(f"{DER}/address/summary.csv"); ad = AD[AD.condition == "name+address+city"]
PV = pd.read_csv(f"{DER}/prompts/summary.csv")
SY = pd.read_csv(f"{DER}/systems/summary.csv")
CL = pd.read_csv(f"{DER}/closed/summary_full.csv"); CN = pd.read_csv(f"{DER}/closed/snowball_full.csv")
TS = pd.read_csv(f"{DER}/ted/summary.csv"); TC = pd.read_csv(f"{DER}/ted/contamination.csv")
RB0 = RB.set_index("model")
CHK.append(("FOPPA, name + city (main)", 6, rng_pp(RB0.loc[MODELS, "dfmr_hi"] * 0 + [C[(C.model == m) & (C.k == 16)].fmr.iloc[0] - C[(C.model == m) & (C.k == 1)].fmr.iloc[0] for m in MODELS]), f"{sig_up(RB0.loc[MODELS, 'dfmr_lo'])}/6"))
CHK.append(("\\quad + street address", 6, rng_pp(ad.dfmr), f"{sig_up(ad.dfmr_lo)}/6"))
for v, lab in [("v1", "\\quad reworded prompt (V1)"), ("v2", "\\quad table + `Answer:' (V2)"), ("v3", "\\quad no entity definition (V3)")]:
    p = PV[PV.variant == v]; CHK.append((lab, 6, rng_pp(p.dfmr_mean), f"{sig_up(p.dfmr_lo)}/6"))
for s, lab in [("batcher", "\\quad BatchER-style batching"), ("comem_select", "\\quad ComEM-style selection"), ("llmcer_cluster", "\\quad LLM-CER-style clustering")]:
    x = SY[(SY.strategy == s) & SY.qtype.isin(NEG)].pivot_table(index=["model", "qtype"], columns="k", values="rate")
    pv = SY[(SY.strategy == s) & SY.qtype.isin(NEG) & (SY.k == 16)].set_index(["model", "qtype"])
    dd = (x[16] - x[1]).groupby(level=0).mean()
    up = pv.discordant_0to1 > pv.discordant_1to0; sig = ((pv.mcnemar_k1_vs_k16_p < .05) & up).groupby(level=0).all()
    CHK.append((lab, len(dd), rng_pp(dd), f"{int(sig.sum())}/{len(dd)}"))
    mac(f"sys{s.split('_')[0].capitalize()}Min", 100 * dd.min(), "{:+.1f}"); mac(f"sys{s.split('_')[0].capitalize()}Max", 100 * dd.max(), "{:+.1f}")
    mac(f"sys{s.split('_')[0].capitalize()}Sig", str(int(sig.sum())))
c16 = CL[CL.k == 16].set_index("model")
c16m = c16.loc[["claude-sonnet-5-nothink", "claude-haiku-5-5"]]
CHK.append(("Claude Sonnet 5, Haiku 5.5", 2, rng_pp(c16m.dfmr_paired), f"{sig_up(c16m.dfmr_lo)}/2"))
CHK.append(("TED CZ/BG, full address", 6, rng_pp(TS.dfmr), f"{sig_up(TS.dfmr_ci_lo)}/6"))
lines = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in CHK]
open(f"{GEN}/tab_checks.tex", "w").write("\n".join(lines) + "\n")
# macros used in the text
for m_, k_ in [("claude-sonnet-5-nothink", "Son"), ("claude-sonnet-5", "SonR"), ("claude-haiku-5-5", "Hai")]:
    r1 = CL[(CL.model == m_) & (CL.k == 1)].iloc[0]; r16 = CL[(CL.model == m_) & (CL.k == 16)].iloc[0]; sn = CN.set_index("model").loc[m_]
    mac(f"cl{k_}Fone", pct(r1.fmr)); mac(f"cl{k_}Fsix", pct(r16.fmr)); mac(f"cl{k_}Rone", pct(r1.recall), "{:.0f}"); mac(f"cl{k_}Rsix", pct(r16.recall), "{:.0f}")
    mac(f"cl{k_}Up", str(int(r16.mc_up))); mac(f"cl{k_}Down", str(int(r16.mc_down))); mac(f"cl{k_}P", sci(r16.mc_p))
    mac(f"cl{k_}Conf", pct(r16.conf_fm)); mac(f"cl{k_}SnC", pct(sn.clean)); mac(f"cl{k_}SnX", pct(sn.contaminated)); mac(f"cl{k_}SnP", f"{sn.p:.3f}")
mac("adMin", 100 * ad.dfmr.min()); mac("adMax", 100 * ad.dfmr.max())
mac("pvSig", str(int(sum(sig_up(PV[PV.variant == v].dfmr_lo) for v in ["v1", "v2", "v3"]))))
mac("tedRecMin", pct(TS.recall_k1.min()), "{:.0f}"); mac("tedRecMax", pct(TS.recall_k1.max()), "{:.0f}")
mac("tedAucMin", TS.auc_k1.min(), "{:.2f}"); mac("tedAucMax", TS.auc_k1.max(), "{:.2f}")
mac("tedDistOneMin", pct(TS.fmr_distinct_k1.min())); mac("tedDistSixMax", pct(TS.fmr_distinct_k16.max()))
mac("tedSevD", 100 * TS.set_index("model").loc["Qwen2.5-7B-Instruct", "dfmr"], "{:+.1f}")
tc = TC; mac("tedSnSig", str(int((tc.p < .05).sum()))); mac("tedSnN", str(len(tc)))
mac("tedSnThirtC", pct(tc[(tc.model == "Qwen2.5-32B-Instruct") & (tc.k == 4)].clean.iloc[0])); mac("tedSnThirtX", pct(tc[(tc.model == "Qwen2.5-32B-Instruct") & (tc.k == 4)].contaminated.iloc[0]))
mac("clSonD", 100 * (CL[(CL.model == "claude-sonnet-5-nothink") & (CL.k == 16)].fmr.iloc[0] - CL[(CL.model == "claude-sonnet-5-nothink") & (CL.k == 1)].fmr.iloc[0]))
mac("clSonRD", 100 * (CL[(CL.model == "claude-sonnet-5") & (CL.k == 16)].fmr.iloc[0] - CL[(CL.model == "claude-sonnet-5") & (CL.k == 1)].fmr.iloc[0]))
mac("clHaiD", 100 * (CL[(CL.model == "claude-haiku-5-5") & (CL.k == 16)].fmr.iloc[0] - CL[(CL.model == "claude-haiku-5-5") & (CL.k == 1)].fmr.iloc[0]))
mac("tedPool", f"{17267:,}")  # distinct (name, address, postcode, town) records with checksum-valid CZ/BG IDs; coverage.csv counts rows (18,712 + 17,607), see data/derived/ted/FINDINGS.md
BQ = pd.read_csv(f"{DER}/systems/batcher_q1_shift.csv").set_index("model")
mac("bqUpThirt", str(int(BQ.loc["Qwen2.5-32B-Instruct", "up"]))); mac("bqDownThirt", str(int(BQ.loc["Qwen2.5-32B-Instruct", "down"])))
mac("recOneMinAll", pct(C[C.k == 1].recall.min()), "{:.0f}"); mac("recOneMaxAll", pct(C[C.k == 1].recall.max()), "{:.0f}")

# ---------------------------------------------------------------- supplementary tables
SH = lambda m: SHORT.get(m, m)
p1 = lambda x: f"{100 * x:.1f}"
import re as _re
MS = lambda t: _re.sub(r"(?<![\w.])-(?=\d)", lambda _m: "\\textminus{}", t)
L = []
for m in MODELS:
    for v in ["v1", "v2", "v3"]:
        r = PV[(PV.model == m) & (PV.variant == v)].iloc[0]
        L.append(f"{SH(m)} & {v.upper()} & {p1(r.fmr_k1)} & {p1(r.fmr_k16)} & {100*r.dfmr_mean:+.1f} [{100*r.dfmr_lo:+.1f}, {100*r.dfmr_hi:+.1f}] & {p1(r.recall_k1)} & {p1(r.recall_k16)} & {r.auc_k1:.2f} & {r.auc_k16:.2f} & {r.yesno_mass_min:.3f} \\\\")
open(f"{GEN}/supp_prompts.tex", "w").write(MS("\n".join(L)) + "\n")
L = []
for _, r in ad.iterrows():
    L.append(f"{SH(r.model)} & {p1(r.fmr1)} & {p1(r.fmr16)} & {100*r.dfmr:+.1f} [{100*r.dfmr_lo:+.1f}, {100*r.dfmr_hi:+.1f}] & {p1(r.dist1)}\\,/\\,{p1(r.dist16)} & {p1(r.or16)} & {p1(r.rec1)} & {p1(r.rec16)} & {r.auc1:.2f} & {r.auc16:.2f} \\\\")
open(f"{GEN}/supp_address.tex", "w").write(MS("\n".join(L)) + "\n")
L = []
for s, lab in [("batcher", "BatchER"), ("comem_select", "ComEM-select"), ("llmcer_cluster", "LLM-CER")]:
    for m in MODELS:
        x = SY[(SY.strategy == s) & (SY.model == m)].set_index(["qtype", "k"])
        if x.empty: continue
        cell = lambda q: f"{p1(x.loc[(q, 1), 'rate'])}\\,$\\rightarrow$\\,{p1(x.loc[(q, 16), 'rate'])}"
        pf = x.parse_fail_rate.mean()
        L.append(f"{lab} & {SH(m)} & {cell('negA')} & {cell('negB')} & {cell('pos')} & ${sci(x.loc[('negA', 16), 'mcnemar_k1_vs_k16_p'])}$ / ${sci(x.loc[('negB', 16), 'mcnemar_k1_vs_k16_p'])}$ & {p1(pf)} \\\\")
open(f"{GEN}/supp_systems.tex", "w").write("\n".join(L) + "\n")
L = []
for _, r in TS.iterrows():
    c4 = TC[(TC.model == r.model) & (TC.k == 4)].iloc[0]
    L.append(f"{SH(r.model)} & {p1(r.fmr_k1)} & {p1(r.fmr_k16)} & {100*r.dfmr:+.1f} [{100*r.dfmr_ci_lo:+.1f}, {100*r.dfmr_ci_hi:+.1f}] & {p1(r.fmr_distinct_k1)}\\,/\\,{p1(r.fmr_distinct_k16)} & {p1(r.fmr_or_k16)} & {p1(r.recall_k1)} & {p1(r.recall_k16)} & {r.auc_k1:.2f} & {p1(c4.clean)}\\,$\\rightarrow$\\,{p1(c4.contaminated)} \\\\")
open(f"{GEN}/supp_ted.tex", "w").write(MS("\n".join(L)) + "\n")
L = []
for m_, lab in [("claude-sonnet-5-nothink", "Sonnet 5, thinking off"), ("claude-haiku-5-5", "Haiku 5.5, thinking off"), ("claude-sonnet-5", "Sonnet 5, default reasoning")]:
    for k in (1, 4, 16):
        r = CL[(CL.model == m_) & (CL.k == k)].iloc[0]
        L.append(f"{lab if k == 1 else ''} & {k} & {p1(r.fmr)} & {p1(r.conf_fm)} & {p1(r.recall)} & {int(r.n_neg)} / {int(r.n_pos)} \\\\")
open(f"{GEN}/supp_closed.tex", "w").write("\n".join(L) + "\n")

L = []
for m in MODELS:
    r = BQ.loc[m]
    L.append(f"{SH(m)} & {r.q1_yes_k1:.1f} & {r.q1_yes_k16:.1f} & {int(r.up)} / {int(r.down)} & ${sci(r.mcnemar_p)}$ \\\\")
open(f"{GEN}/supp_batcherq.tex", "w").write("\n".join(L) + "\n")
TV = pd.read_csv(f"{DER}/ted/coverage.csv")
L = []
for _, r in TV[TV.id_scheme.notna()].iterrows():
    scheme = r.id_scheme.split(" (")[0].replace("ICO", "I\\v{C}O")
    ne = "--" if pd.isna(r.entities_ge17_records) else str(int(r.entities_ge17_records))
    L.append(f"{r.country_code} & {scheme} & {int(r.win_records_total):,} & {r.nid_coverage_pct:.1f} & {r.valid_rate_pct:.1f} & {ne} \\\\")
open(f"{GEN}/supp_tedcov.tex", "w").write("\n".join(L).replace(",", "{,}") + "\n")
# ---------------------------------------------------------------- fix track (online linking operating points)
FX = f"{DER}/fix"
CVO = pd.read_csv(f"{FX}/sweep/cv_operating_point.csv"); MRG = pd.read_csv(f"{FX}/sweep/matched_recall_gains.csv")
F1D = pd.read_csv(f"{FX}/deltas.csv"); FS = pd.read_csv(f"{FX}/summary.csv"); SWS = pd.read_csv(f"{FX}/sweep/sweep_summary.csv")
SOFT = pd.read_csv(f"{FX}/sweep/audit_soft_vs_profile.csv")
mac("fxDPmin", 100 * CVO.d_precision.min(), "{:+.1f}"); mac("fxDPmax", 100 * CVO.d_precision.max(), "{:+.1f}")
gain = CVO[CVO.d_recall_lo > 0]
mac("fxRsig", str(len(gain))); mac("fxRRmin", gain.recall_ratio.min(), "{:.1f}"); mac("fxRRmax", gain.recall_ratio.max(), "{:.1f}")
mac("fxPsigNone", str(int(((CVO.d_precision_lo > 0) | (CVO.d_precision_hi < 0)).sum())))
mac("fxQthreeRatio", CVO.set_index("model").loc["Qwen3-32B", "recall_ratio"], "{:.2f}")
dm = MRG.dropna(subset=["gain_vs_profile"])
mac("fxCalN", str(len(dm))); mac("fxCalSig", str(int(((dm.gain_vs_profile_lo > 0) | (dm.gain_vs_profile_hi < 0)).sum())))
au = F1D[F1D.rep == "profile_audit"]
mac("fxAudProf", au[au.vs == "profile"].d_f1.mean(), "{:+.2f}"); mac("fxAudAnc", au[au.vs == "anchor"].d_f1.mean(), "{:+.2f}")
mac("fxAudAncSig", str(int((au[au.vs == "anchor"].d_f1_lo > 0).sum())))
sa = SWS[SWS.variant == "profile_audit_soft"]
mac("fxSoftEvMin", str(int(sa.n_evictions.min()))); mac("fxSoftEvMax", str(int(sa.n_evictions.max())))
mac("fxSoftDP", 100 * SOFT.d_precision.abs().max(), "{:.1f}")
cpr = FS.pivot_table(index="model", columns="rep", values="calls_per_record").mean()
mac("fxCallsAnc", cpr["anchor"], "{:.2f}"); mac("fxCallsProf", cpr["profile"], "{:.2f}"); mac("fxCallsAud", cpr["profile_audit"], "{:.2f}")
L = []
for m in MODELS:
    r = CVO.set_index("model").loc[m]
    L.append(f"{SH(m)} & {p1(r.anchor_precision)} / {p1(r.anchor_recall)} & {p1(r.profile_precision)} / {p1(r.profile_recall)} & "
             f"{100*r.d_precision:+.1f} [{100*r.d_precision_lo:+.1f}, {100*r.d_precision_hi:+.1f}] & {r.recall_ratio:.2f} & {100*r.d_recall:+.1f} [{100*r.d_recall_lo:+.1f}, {100*r.d_recall_hi:+.1f}] \\\\")
open(f"{GEN}/supp_fix_cv.tex", "w").write(MS("\n".join(L)) + "\n")
L = []
labs = {"anchor": "founding member", "profile": "whole profile", "profile_cal": "size-calibrated", "profile_audit": "profile + audit", "profile_cal_audit": "calibrated + audit"}
for m in MODELS:
    for rep_, lab in labs.items():
        r = FS[(FS.model == m) & (FS.rep == rep_)].iloc[0]
        L.append(f"{SH(m) if rep_ == 'anchor' else ''} & {lab} & {p1(r.precision)} & {p1(r.recall)} & {p1(r.f1)} & {p1(r.mixed_share)} & {r.calls_per_record:.2f} \\\\")
    r = sa[sa.model == m].iloc[0]
    L.append(f" & soft audit & {p1(r.precision)} & {p1(r.recall)} & {p1(r.f1)} & {p1(r.mixed_share)} & {r.calls_per_record:.2f} \\\\")
open(f"{GEN}/supp_fix_reps.tex", "w").write("\n".join(L) + "\n")
# supplementary figure: precision-recall operating points
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as _plt
PRP = pd.read_csv(f"{FX}/sweep/pr_points.csv")
fig, axes = _plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=False, sharey=True)
sty = {"anchor_global": ("founding member, global $t$", "#7f7f7f", "s"), "profile_global": ("whole profile, global $t$", "#1f77b4", "o"), "profile_cal": ("whole profile, size-calibrated $\\tau$", "#d62728", "^")}
for ax, m in zip(axes.ravel(), MODELS):
    for fam, (lab, col, mk) in sty.items():
        x = PRP[(PRP.model == m) & (PRP.family == fam) & (PRP.recall >= 0.002)].dropna(subset=["precision"]).sort_values("recall")
        ax.plot(100 * x.recall, 100 * x.precision, marker=mk, color=col, ms=3.5, lw=1, label=lab)
    r = CVO.set_index("model").loc[m]
    ax.plot(100 * r.profile_recall, 100 * r.profile_precision, marker="*", color="black", ms=8, ls="none", label="cross-validated profile point")
    ax.set_title(SH(m), fontsize=8); ax.tick_params(labelsize=7); ax.set_ylim(55, 100)
for ax in axes[1]: ax.set_xlabel("pairwise recall (%)", fontsize=8)
for ax in axes[:, 0]: ax.set_ylabel("pairwise precision (%)", fontsize=8)
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{FIG}/figS1_operating_points.pdf"); fig.savefig(f"{FIG}/figS1_operating_points.png", dpi=200)
# ---------------------------------------------------------------- robustness checks, set 1
RV = f"{DER}/review"
NAB = pd.read_csv(f"{RV}/neg_ab.csv"); na = NAB[NAB.neg == "negA"]; nb_ = NAB[NAB.neg == "negB"]
mac("negAoneMin", pct(na.fmr_k1.min())); mac("negAoneMax", pct(na.fmr_k1.max())); mac("negAsixMin", pct(na.fmr_k16.min())); mac("negAsixMax", pct(na.fmr_k16.max()))
mac("negBsixMin", pct(nb_.fmr_k16.min())); mac("negBsixMax", pct(nb_.fmr_k16.max())); mac("negABpMax", sci(NAB.p.max()))
mac("entTieThirt", str(int(len(ENT) - RB.set_index("model").loc["Qwen2.5-32B-Instruct", "n_ent_up"] - RB.set_index("model").loc["Qwen2.5-32B-Instruct", "n_ent_down"])))
TB = pd.read_csv(f"{RV}/tfidf_baseline.csv").set_index("input")
mac("tfAucOne", TB.loc["name", "auc_k1"], "{:.2f}"); mac("tfAucSix", TB.loc["name", "auc_k16max"], "{:.2f}")
HS = pd.read_csv(f"{RV}/similar_negatives.csv"); hf = HS[HS.dataset == "FOPPA"]; ht = HS[HS.dataset == "TED"]
mac("hsFn", str(int(hf.n.iloc[0]))); mac("hsTn", str(int(ht.n.iloc[0])))
mac("hsFoneMin", pct(hf.fmr_k1.min())); mac("hsFoneMax", pct(hf.fmr_k1.max())); mac("hsFsixMin", pct(hf.fmr_k16.min())); mac("hsFsixMax", pct(hf.fmr_k16.max()))
mac("hsToneMin", pct(ht.fmr_k1.min())); mac("hsToneMax", pct(ht.fmr_k1.max())); mac("hsTsixMin", pct(ht.fmr_k16.min())); mac("hsTsixMax", pct(ht.fmr_k16.max()))
mac("hsTsigUp", str(int(((ht.p < .05) & (ht.up > ht.down)).sum())))
NH = pd.read_csv(f"{RV}/neg_hardness.csv").set_index(["dataset", "neg"])
mac("hardF", NH.loc[("FOPPA", "negA"), "median"], "{:.2f}"); mac("hardT", NH.loc[("TED", "negA"), "median"], "{:.2f}")
SO = pd.read_csv(f"{RV}/saturation_online.csv").set_index("model")
mac("satOnThirt", pct(SO.loc["Qwen2.5-32B-Instruct", "share_saturated"]), "{:.0f}"); mac("satOnFourt", pct(SO.loc["Qwen2.5-14B-Instruct", "share_saturated"]), "{:.0f}")
mac("satOnThirtWrong", pct(SO.loc["Qwen2.5-32B-Instruct", "sat_incorrect"]), "{:.0f}")
BCU = pd.read_csv(f"{RV}/bcubed.csv"); bp = BCU.pivot(index="model", columns="rep")
bc_best = bp["bc_f1"][["anchor", "pairmax", "profile"]].idxmax(axis=1)
mac("bcProfBest", str(int((bc_best == "profile").sum())))
mac("bcProfOverAnc", str(int((bp["bc_f1"]["profile"] > bp["bc_f1"]["anchor"]).sum())))
mac("bcCvRecUp", str(int((bp["bc_recall"]["cv_profile_point"] > bp["bc_recall"]["anchor"]).sum())))
dbp = 100 * (bp["bc_precision"]["cv_profile_point"] - bp["bc_precision"]["anchor"])
mac("bcCvDPmin", dbp.min(), "{:+.1f}"); mac("bcCvDPmax", dbp.max(), "{:+.1f}")
L = []
for m in MODELS:
    a_, b2 = na.set_index("model").loc[m], nb_.set_index("model").loc[m]
    L.append(f"{SH(m)} & {p1(a_.fmr_k1)}\\,$\\rightarrow$\\,{p1(a_.fmr_k16)} & {int(a_.up)} / {int(a_.down)} & {p1(b2.fmr_k1)}\\,$\\rightarrow$\\,{p1(b2.fmr_k16)} & {int(b2.up)} / {int(b2.down)} \\\\")
open(f"{GEN}/supp_negab.tex", "w").write("\n".join(L) + "\n")
L = []
labs_bc = {"anchor": "founding member", "pairmax": "pairwise max", "profile": "whole profile", "cv_profile_point": "precision-matched profile"}
for m in MODELS:
    for rep_, lab in labs_bc.items():
        r = bp.loc[m]
        L.append(f"{SH(m) if rep_ == 'anchor' else ''} & {lab} & {p1(r[('bc_precision', rep_)])} & {p1(r[('bc_recall', rep_)])} & {p1(r[('bc_f1', rep_)])} \\\\")
open(f"{GEN}/supp_bcubed.tex", "w").write("\n".join(L) + "\n")
L = []
for _, r in HS.iterrows():
    L.append(f"{r.dataset if r.model == MODELS[0] else ''} & {SH(r.model)} & {int(r.n)} & {p1(r.fmr_k1)}\\,$\\rightarrow$\\,{p1(r.fmr_k16)} & {int(r.up)} / {int(r.down)} & ${sci(r.p)}$ \\\\")
open(f"{GEN}/supp_simneg.tex", "w").write("\n".join(L) + "\n")
LN = pd.read_csv(f"{DER}/length/summary.csv").set_index("model")
mac("lnDown", str(int(((LN.rep16_p < .05) & (LN.rep16_down > LN.rep16_up)).sum())))
mac("lnThirtOne", pct(LN.loc["Qwen2.5-32B-Instruct", "fmr_set1"])); mac("lnThirtRep", pct(LN.loc["Qwen2.5-32B-Instruct", "fmr_rep16"])); mac("lnThirtSet", pct(LN.loc["Qwen2.5-32B-Instruct", "fmr_set16"]))
mac("lnMisOne", pct(LN.loc["Mistral-Small-24B-Instruct-2501", "fmr_set1"])); mac("lnMisRep", pct(LN.loc["Mistral-Small-24B-Instruct-2501", "fmr_rep16"]))
mac("lnSetMin", pct(LN.fmr_set16.min())); mac("lnSetMax", pct(LN.fmr_set16.max()))
L = []
for m in MODELS:
    r = LN.loc[m]
    L.append(f"{SH(m)} & {p1(r.fmr_set1)} & {p1(r.fmr_rep4)} & {p1(r.fmr_rep16)} & {p1(r.fmr_set16)} & {int(r.rep16_up)} / {int(r.rep16_down)} & ${sci(r.rep16_p)}$ & {p1(r.rec_set1)} / {p1(r.rec_rep16)} / {p1(r.rec_set16)} \\\\")
open(f"{GEN}/supp_length.tex", "w").write("\n".join(L) + "\n")
# ---------------------------------------------------------------- robustness checks, set 2
CALLS = pd.read_csv(f"{RV}/calls_online.csv").pivot(index="model", columns="rep", values="calls_per_record")
mac("callPmMin", CALLS.pairmax.min(), "{:.1f}"); mac("callPmMax", CALLS.pairmax.max(), "{:.1f}")
mac("callPrMin", CALLS.profile.min(), "{:.1f}"); mac("callPrMax", CALLS.profile.max(), "{:.1f}")
HO = pd.read_csv(f"{RV}/holm_table3.csv"); mac("holmN", str(len(HO))); mac("holmSame", "all" if (HO.sig_up_raw == HO.sig_up_holm).all() else "not all")
CVO2 = pd.read_csv(f"{FX}/sweep/cv_operating_point.csv")
mac("fxNiMin", -100 * CVO2.d_precision_lo90.max(), "{:.1f}"); mac("fxNiMax", -100 * CVO2.d_precision_lo90.min(), "{:.1f}")
mis = CVO2.set_index("model").loc["Mistral-Small-24B-Instruct-2501"]
mac("fxMisRone", pct(mis.anchor_recall)); mac("fxMisRtwo", pct(mis.profile_recall))
CT = pd.read_csv(f"{RV}/conf_thresholds.csv").set_index(["model", "cut"])
mac("cfNineFiveThirt", pct(CT.loc[("Qwen2.5-32B-Instruct", .95), "profile"])); mac("cfNineFiveBestThirt", pct(CT.loc[("Qwen2.5-32B-Instruct", .95), "best_member"]))
mac("cfTripThirt", pct(CT.loc[("Qwen2.5-32B-Instruct", .999), "profile"])); mac("cfTripBestThirt", pct(CT.loc[("Qwen2.5-32B-Instruct", .999), "best_member"]))
L = []
for m in MODELS:
    L.append(f"{SH(m)} & {CALLS.loc[m, 'anchor']:.2f} & {CALLS.loc[m, 'pairmax']:.2f} & {CALLS.loc[m, 'profile']:.2f} \\\\")
open(f"{GEN}/supp_calls.tex", "w").write("\n".join(L) + "\n")
TC2 = pd.read_csv(f"{RV}/ted_by_country.csv")
L = []
for m in MODELS:
    x = TC2[TC2.model == m].set_index("country")
    L.append(f"{SH(m)} & " + " & ".join(f"{p1(x.loc[c, 'fmr_k1'])}\\,$\\rightarrow$\\,{p1(x.loc[c, 'fmr_k16'])} & {int(x.loc[c, 'up'])} / {int(x.loc[c, 'down'])}" for c in ["CZ", "BG"]) + " \\\\")
open(f"{GEN}/supp_tedcountry.tex", "w").write("\n".join(L) + "\n")
LOS_ = pd.read_csv(f"{DER}/logodds/summary.csv").set_index("model"); lt = LOS_.loc["Qwen2.5-32B-Instruct"]
mac("loRoneThirt", pct(lt.cv_rec1)); mac("loRsixThirt", pct(lt.cv_rec16)); mac("loSatThirt", str(int(lt.sat16))); mac("loSatDistThirt", str(int(lt.sat16_distinct_lo)))
mac("loLoMin", lt.sat16_lo_min, "{:.0f}"); mac("loLoMax", lt.sat16_lo_max, "{:.0f}"); mac("loAgree", pct(LOS_.agree_with_orig.min()))
mac("loAucThirt", lt.auc16_lo, "{:.2f}")
L = []
for m in LOS_.index:
    r = LOS_.loc[m]
    L.append(f"{SH(m)} & {r.agree_with_orig*100:.1f} & {int(r.sat16)} & {int(r.sat16_distinct_lo)} & {r.auc16_p:.3f} / {r.auc16_lo:.3f} & {p1(r.cv_rec1)} / {p1(r.cv_rec16)} \\\\")
open(f"{GEN}/supp_logodds.tex", "w").write("\n".join(L) + "\n")
for m_, k_ in [("claude-sonnet-5-nothink", "Son"), ("claude-sonnet-5", "SonR"), ("claude-haiku-5-5", "Hai")]:
    r16_ = CL[(CL.model == m_) & (CL.k == 16)].iloc[0]; mac(f"cl{k_}Lo", 100 * r16_.dfmr_lo, "{:.1f}"); mac(f"cl{k_}Hi", 100 * r16_.dfmr_hi, "{:.1f}")
# ---------------------------------------------------------------- dataset macros
mac("nSupplier", f"{stats['supplier_agents']:,}"); mac("nSiret", f"{stats['with_valid_siret']:,}")
mac("pSiret", pct(stats["with_valid_siret"] / stats["supplier_agents"])); mac("nSiren", f"{stats['distinct_siren']:,}")
mac("nEnt", str(stats["siren_ge17"])); mac("nPrompts", f"{16758:,}"); mac("nMitPrompts", f"{stats['mitigation_prompts']:,}")

with open(f"{GEN}/numbers.tex", "w") as f:
    f.write("% generated by scripts/build_assets.py -- do not edit\n")
    for k, v in sorted(macros.items()): f.write(f"\\newcommand{{\\{k}}}{{{v.replace(',', '{,}')}}}\n")

# ---------------------------------------------------------------- tables
f1 = lambda x: f"{pct(x):.1f}"
SEP = "\\,/\\,"
LOS = pd.read_csv(f"{DER}/logodds/summary.csv").set_index("model")
mcell = lambda r: (f1(r.rec1) + SEP + f1(r.rec16)) if r.attain > 0 else ((f1(LOS.loc[r.name, "cv_rec1"]) + SEP + f1(LOS.loc[r.name, "cv_rec16"]) + "$^\\dagger$") if r.name in LOS.index else f1(r.rec1) + SEP + "n/a")
lines = []
for m in MODELS:
    c = C[C.model == m].set_index("k"); cf = CF.set_index("model").loc[m]; mf = MF[MF.model == m].set_index("k"); cvr = CV.set_index("model").loc[m]
    lines.append(f"{SHORT[m]} & {f1(c.loc[1,'fmr'])} & {f1(c.loc[16,'fmr'])} & {f1(c.loc[16,'fmr_or'])} & "
                 f"{f1(c.loc[1,'fmr_distinct'])}\\,/\\,{f1(c.loc[16,'fmr_distinct'])} & {f1(c.loc[1,'recall'])} & {f1(c.loc[16,'recall'])} & "
                 f"{c.loc[1,'auc']:.2f} & {c.loc[16,'auc']:.2f} & {f1(cf.best_single16)} & {f1(cf.profile16)} & "
                 + mcell(cvr) + " \\\\")
open(f"{GEN}/tab_main.tex", "w").write("\n".join(lines) + "\n")
lines = []
for m in MODELS:
    e = E[E.model == m].set_index("rep"); c4 = CO[(CO.model == m) & (CO.k == 4)].iloc[0]; c16 = CO[(CO.model == m) & (CO.k == 16)].iloc[0]
    lines.append(f"{SHORT[m]} & {f1(c4.clean)} & {f1(c4.contaminated)} & {f1(c16.clean)} & {f1(c16.contaminated)} & "
                 + " & ".join(f"{e.loc[r,'precision']:.2f} / {e.loc[r,'recall']:.2f} / {e.loc[r,'f1']:.2f}" for r in ["anchor", "pairmax", "profile"]) + " \\\\")
open(f"{GEN}/tab_snow_e2e.tex", "w").write("\n".join(lines) + "\n")

lines = []
for m in MODELS:
    r = RB.set_index("model").loc[m]; g_ = SG.set_index("model").loc[m]; c = C[C.model == m].set_index("k")
    lines.append(f"{SHORT[m]} & {pct(c.loc[16,'fmr']-c.loc[1,'fmr']):.1f} [{pct(r.dfmr_lo):.1f}, {pct(r.dfmr_hi):.1f}] & "
                 f"{c.loc[16,'auc']-c.loc[1,'auc']:.2f} [{r.dauc_lo:.2f}, {r.dauc_hi:.2f}] & {int(r.n_ent_up)}/{int(r.n_ent_down)} & ${sci(r.p_entity)}$ & "
                 f"{f1(g_.fmr_founding)} / {f1(g_.fmr_other_mean)} & ${r.rho_size:+.2f}$ \\\\")
open(f"{GEN}/tab_robust.tex", "w").write("\n".join(lines) + "\n")
lines = []
for m in MODELS:
    mm = MI[MI.model == m].set_index("variant")
    lines.append(f"{SHORT[m]} & " + " & ".join(f"{f1(mm.loc[v,'fmr'])} / {f1(mm.loc[v,'recall'])}" for v in
                 ["founding member only", "whole profile (k=16)", "skeptical prompt (k=16)", "last 4 members", "founding + last 3"]) + " \\\\")
open(f"{GEN}/tab_mitig.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix"})
def letter(ax, s): ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
def logk(ax):
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
    ax.set_xlim(0.85, 19); ax.set_xlabel("profile size $k$ (verified members)")

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.05]})
ax = axes[0]
for m in MODELS:
    c = C[C.model == m]; ax.plot(c.k, c.fmr, marker=MKM[m], ls="-", color=COL[m], ms=3.2, lw=1.2, label=SHORT[m])
    ax.plot(c.k, c.fmr_or, ":", color=COL[m], lw=1.0)
ax.set_ylim(0, 0.5); ax.set_ylabel("share of hard negatives\njudged the same entity"); logk(ax)
ax.set_title("False matches rise with profile size", loc="left"); letter(ax, "a")
ax.text(0.03, 0.97, "dotted: OR of the model's own\nsingle-member judgments", transform=ax.transAxes, ha="left", va="top", fontsize=6, color="0.35")
ax = axes[1]
for m in MODELS:
    c = C[C.model == m]; ax.plot(c.k, c.recall, marker=MKM[m], ls="-", color=COL[m], ms=3.2, lw=1.2)
ax.set_ylim(0, 1); ax.set_ylabel("share of held-out true members\njudged the same entity"); logk(ax)
ax.set_title("So does recall", loc="left"); letter(ax, "b")
handles = [plt.Line2D([], [], marker=MKM[m], ls="-", color=COL[m], ms=3.2, lw=1.2, label=SHORT[m]) for m in MODELS]
ax = axes[2]; x = np.arange(len(MODELS)); w = .26
ax.bar(x - w, CF.k1, w, color="0.82", label="$k$=1 (founding member)")
ax.bar(x, CF.best_single16, w, color="0.5", label="most similar single member, $k$=16")
ax.bar(x + w, CF.profile16, w, color="#a50f15", label="whole profile, $k$=16")
ax.set_xticks(x); ax.set_xticklabels([SHORT[m].replace("Qwen", "Q").replace("OLMo-2-", "OLMo-").replace("Mistral-", "Mis-") for m in MODELS], rotation=30, ha="right", rotation_mode="anchor"); ax.set_ylim(0, 0.52)
ax.set_ylabel("share of hard negatives\nwith P(YES) > 0.99"); ax.set_title("Confident false matches", loc="left"); letter(ax, "c")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1)); fig.legend(handles=handles, loc="lower center", ncol=len(MODELS), frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.2)
fig.savefig(f"{FIG}/fig1_accretion.pdf"); fig.savefig(f"{FIG}/fig1_accretion.png", dpi=300); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
MK = {"founding member only": "o", "whole profile (k=16)": "s", "skeptical prompt (k=16)": "^", "last 4 members": "v", "founding + last 3": "D"}
for m in MODELS:
    c = C[C.model == m]; ax.plot(c.fmr, c.recall, ls=LSM[m], color=COL[m], lw=1.1, alpha=.9, zorder=1)
    for v, mk in MK.items():
        r = MI[(MI.model == m) & (MI.variant == v)].iloc[0]
        ax.scatter(r.fmr, r.recall, marker=mk, s=20, color=COL[m], edgecolor="k", lw=.3, zorder=3)
for v, mk in MK.items(): ax.scatter([], [], marker=mk, s=20, color="0.6", edgecolor="k", lw=.3, label=v)
for m in MODELS: ax.plot([], [], ls=LSM[m], color=COL[m], lw=1.1, label=SHORT[m])
ax.set_xlim(0, 0.48); ax.set_ylim(0, 1); ax.set_xlabel("false-match rate on hard negatives"); ax.set_ylabel("recall of held-out true members")
ax.set_title("Mitigations stay near the accretion curve (lines: $k$=1→16)", loc="left"); letter(ax, "a")
ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=5.8, handlelength=2.2, columnspacing=0.8)
ax = axes[1]; x = np.arange(len(MODELS)); w = .36
c4 = CO[CO.k == 4].set_index("model").loc[MODELS]
ax.bar(x - w / 2, c4.clean, w, color="0.6", label="4 verified members")
ax.bar(x + w / 2, c4.contaminated, w, color="#a50f15", label="3 verified members + 1 false merge")
ax.set_xticks(x); ax.set_xticklabels([SHORT[m].replace("Qwen", "Q").replace("OLMo-2-", "OLMo-").replace("Mistral-", "Mis-") for m in MODELS]); ax.set_ylim(0, 0.8)
ax.set_ylabel("share judged the same entity"); ax.set_title("One false merge attracts the intruder's kin", loc="left"); letter(ax, "b")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(w_pad=1.5); fig.savefig(f"{FIG}/fig2_mitigation_snowball.pdf"); fig.savefig(f"{FIG}/fig2_mitigation_snowball.png", dpi=300); plt.close(fig)
print("macros", len(macros))
