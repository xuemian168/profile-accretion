"""Operating-curve sweep for end-to-end online linking: global-threshold profile/anchor curves,
size-calibrated profile_cal curves (threshold fit directly to a target hard-negative FMR tau,
including the k=1 bin), and a softer two-sided audit variant.

A variant spec (one dict per line of --variants, a JSON list) is one of:
  {"name": "profile_t0.5",        "use_profile": true,  "threshold": 0.5}                         # global
  {"name": "anchor_t0.95",        "use_profile": false, "threshold": 0.95}                        # global
  {"name": "profile_cal_tau0.10", "use_profile": true,  "cal_bins": {"1":.., "2":.., "4":.., "8":.., "16":..}}
  {"name": "profile_audit_soft",  "use_profile": true,  "threshold": 0.5,
   "audit_soft": {"milestones": [4,8,16], "founder_t": 0.05, "loo_t": 0.5}}

Loads the model ONCE and runs every variant in --variants over the same 80 blocks, appending all
rows (tagged with a "variant" field) to one output jsonl.

python e2e_sweep.py --blocks blocks_e2e.json --model Qwen/Qwen2.5-32B-Instruct \
    --variants variants.json --out sweep_<tag>.jsonl
"""
import argparse, json, math, time
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYS = ("You are an entity-resolution expert for French public-procurement supplier records. "
       "Each record is 'NAME | CITY POSTCODE'. Two records refer to the same legal entity if they share the same SIREN "
       "(different establishments/branches of one company count as the SAME). Distinct legal entities in the same corporate "
       "group (subsidiaries, sister companies, franchisees) count as DIFFERENT.")
MAXM = 16


def user_prompt(members, q):
    lst = "\n".join(f"{i+1}. {m}" for i, m in enumerate(members))
    return (f"Known entity — the following {len(members)} record(s) were verified to belong to ONE legal entity:\n{lst}\n\n"
            f"New record: {q}\n\nDoes the new record belong to that same legal entity? Answer with exactly one word: YES or NO.")


def grams(t):
    s = " " + t.split(" | ")[0] + " "
    return Counter(s[i:i + 3] for i in range(len(s) - 2))


def cos(a, b):
    num = sum(v * b.get(k, 0) for k, v in a.items())
    return num / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())) + 1e-9)


def shown_excl(members, excl_idx=None):  # founding member + most recent ones, at most MAXM, optionally dropping one index
    idx = [i for i in range(len(members)) if i != excl_idx]
    if len(idx) <= MAXM:
        keep = idx
    else:
        keep = [idx[0]] + idx[-(MAXM - 1):]
    return [members[i] for i in keep]


def bin_k(size):
    if size <= 1: return "1"
    if size == 2: return "2"
    if size <= 4: return "4"
    if size <= 8: return "8"
    return "16"


def threshold_for(variant, size_before):
    if "cal_bins" in variant:
        return variant["cal_bins"][bin_k(size_before)]
    return variant["threshold"]


def audit_soft(c, score, spec):
    """Evict member j (j>0) only if BOTH founder-alone P(YES) < founder_t AND the leave-one-out
    profile (rest of the cluster, capped at MAXM) P(YES) < loo_t. Returns (evicted_out, scored_out, n_calls)."""
    n = len(c["members"])
    if n < 2:
        return [], [], 0
    founder = [c["members"][0]]
    idx = list(range(1, n))
    pairs_f = [(founder, c["members"][j]) for j in idx]
    pairs_loo = [(shown_excl(c["members"], excl_idx=j), c["members"][j]) for j in idx]
    ps_f = score(pairs_f); ps_loo = score(pairs_loo)
    keep_idx, evicted, scored_out = [0], [], []
    for j, pf, pl in zip(idx, ps_f, ps_loo):
        evict = (pf < spec["founder_t"]) and (pl < spec["loo_t"])
        scored_out.append({"siren": c["sirens"][j], "p_founder": round(float(pf), 4), "p_loo": round(float(pl), 4), "kept": not evict})
        if evict:
            evicted.append((j, pf, pl))
        else:
            keep_idx.append(j)
    orig_members, orig_grams, orig_sirens = c["members"], c["grams"], c["sirens"]
    c["members"] = [orig_members[j] for j in keep_idx]
    c["grams"] = [orig_grams[j] for j in keep_idx]
    c["sirens"] = [orig_sirens[j] for j in keep_idx]
    evicted_out = [(orig_members[j], orig_sirens[j], orig_grams[j], float(pf), float(pl)) for j, pf, pl in evicted]
    return evicted_out, scored_out, len(pairs_f) + len(pairs_loo)


def run_variant(blocks, variant, score, f, t0):
    name = variant["name"]
    use_profile = variant["use_profile"]
    audit_spec = variant.get("audit_soft")
    ncalls = 0
    for b in blocks:
        clusters = []  # each: {"members": [text], "grams": [Counter], "sirens": [true siren]}
        for r_i, rec in enumerate(b["records"]):
            g = grams(rec["text"])
            cand = sorted(((max(cos(g, h) for h in c["grams"]), ci) for ci, c in enumerate(clusters)), reverse=True)
            cand = [ci for s, ci in cand[:3] if s >= 0.3]
            best, bestp, cscores = None, 0.0, {}
            if cand:
                pairs = [((shown_excl(clusters[ci]["members"]) if use_profile else [clusters[ci]["members"][0]]), rec["text"]) for ci in cand]
                ps = score(pairs); ncalls += len(pairs)
                cscores = dict(zip(cand, ps))
                best, bestp = max(cscores.items(), key=lambda kv: kv[1])
            threshold_used = 0.5
            joined = False
            size_before = 0
            if best is not None:
                size_before = len(clusters[best]["members"])
                threshold_used = threshold_for(variant, size_before)
                joined = bestp > threshold_used
            if joined:
                c = clusters[best]
                c["members"].append(rec["text"]); c["grams"].append(g); c["sirens"].append(rec["siren"]); cid = best
                new_size = len(c["members"])
                if audit_spec and new_size in set(audit_spec["milestones"]):
                    evicted_out, scored_out, n_audit_calls = audit_soft(c, score, audit_spec)
                    ncalls += n_audit_calls
                    audits_this_record = []
                    for ev_text, ev_siren, ev_grams, ev_pf, ev_pl in evicted_out:
                        clusters.append({"members": [ev_text], "grams": [ev_grams], "sirens": [ev_siren]})
                        audits_this_record.append({"evicted_siren": ev_siren, "p_founder": round(ev_pf, 4), "p_loo": round(ev_pl, 4), "new_cluster": len(clusters) - 1})
                    f.write(json.dumps({"variant": name, "block": b["block"], "i": r_i, "audit": True,
                                        "trigger_cluster": cid, "milestone_size": new_size,
                                        "n_scored": n_audit_calls, "scored": scored_out,
                                        "n_evicted": len(evicted_out), "evictions": audits_this_record}) + "\n")
            else:
                clusters.append({"members": [rec["text"]], "grams": [g], "sirens": [rec["siren"]]}); cid = len(clusters) - 1
            f.write(json.dumps({"variant": name, "block": b["block"], "i": r_i, "siren": rec["siren"], "cluster": cid,
                                "joined": joined, "size_before": size_before, "p_best": bestp,
                                "threshold_used": round(threshold_used, 4),
                                "n_cand": len(cand), "cand_scores": {str(k): round(v, 4) for k, v in cscores.items()}}) + "\n")
        f.flush()
    print("variant", name, "done", round(time.time() - t0), "s calls", ncalls, flush=True)
    return ncalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=24)
    a = ap.parse_args()
    blocks = json.load(open(a.blocks))
    variants = json.load(open(a.variants))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        print("fallback load", repr(e)[:200], flush=True)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    def ids_for(ws):
        s = set()
        for w in ws:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t: s.add(t[0])
        return sorted(s)
    YES = ids_for(["YES", "Yes", "yes"]); NO = ids_for(["NO", "No", "no"])
    kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}

    @torch.inference_mode()
    def score(pairs):
        out = []
        for i in range(0, len(pairs), a.bs):
            texts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user_prompt(m, q)}],
                                             tokenize=False, add_generation_prompt=True, **kw) for m, q in pairs[i:i + a.bs]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            try:
                lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                lg = model(**enc).logits[:, -1, :].float()
            pr = torch.softmax(lg, -1); py = pr[:, YES].sum(-1); pn = pr[:, NO].sum(-1)
            out += (py / (py + pn + 1e-12)).tolist()
        return out

    f = open(a.out, "w")
    for variant in variants:
        run_variant(blocks, variant, score, f, t0)
    f.close()
    print("FINISHED", a.model, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
