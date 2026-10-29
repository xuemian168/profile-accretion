"""End-to-end online linking with two practical fixes on top of the plain-profile representation.

Baselines (re-run with this same code for comparability): 'anchor' (compare new record to the
founding member only, threshold 0.5), 'profile' (compare to the whole shown profile, threshold 0.5).

Fixes:
  'profile_cal'      whole profile, but the accept threshold depends on the current size of the
                      candidate cluster via the bins fit in fit_thresholds.py (size 1->t_1=0.5,
                      2->t_2, 3-4->t_4, 5-8->t_8, 9-16(+)->t_16).
  'profile_audit'     whole profile, threshold fixed at 0.5, PLUS: whenever a cluster's size just
                      reached 4, 8 or 16 members, re-score every non-founding member against the
                      founding member ALONE and evict members with P(YES) < 0.5 into their own new
                      singleton clusters.
  'profile_cal_audit' both: calibrated threshold for the join decision + the size-4/8/16 audit.

python e2e_fix.py --blocks blocks_e2e.json --model Qwen/Qwen2.5-32B-Instruct \
    --thresholds thresholds.json --reps anchor,profile,profile_cal,profile_audit,profile_cal_audit --out e2e_fix_<tag>.jsonl
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
AUDIT_SIZES = {4, 8, 16}
CAL_REPS = {"profile_cal", "profile_cal_audit"}
AUDIT_REPS = {"profile_audit", "profile_cal_audit"}
PROFILE_REPS = {"profile", "profile_cal", "profile_audit", "profile_cal_audit"}


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


def shown(members):  # founding member + most recent ones, at most MAXM
    return members if len(members) <= MAXM else [members[0]] + members[-(MAXM - 1):]


def bin_k(size):  # map current (pre-join) cluster size to the fitted bin key
    if size <= 1: return "1"
    if size == 2: return "2"
    if size <= 4: return "4"
    if size <= 8: return "8"
    return "16"


def load_thresholds(path, model_hf_id):
    d = json.load(open(path))
    key = model_hf_id.split("/")[-1]
    if key not in d:
        raise KeyError(f"no thresholds for model key {key!r} in {path} (keys: {[k for k in d if k != '_meta']})")
    return {k: v["t"] for k, v in d[key]["bins"].items()}


def audit_cluster(c, score, milestone_size):
    """Re-score every non-founding member of cluster dict c against the founding member alone.
    Returns (evicted_out, scored_out, n_calls) where scored_out lists (siren, p, kept) for every
    member checked (so the audit is fully traceable even when nothing is evicted) and evicted_out
    is the subset that is kept out, with enough to seed a fresh singleton cluster."""
    n = len(c["members"])
    if n < 2:
        return [], [], 0
    founder = [c["members"][0]]
    idx = list(range(1, n))
    pairs = [(founder, c["members"][j]) for j in idx]
    ps = score(pairs)
    keep_idx, evicted, scored_out = [0], [], []
    for j, p in zip(idx, ps):
        kept = p >= 0.5
        scored_out.append({"siren": c["sirens"][j], "p": round(float(p), 4), "kept": kept})
        if kept:
            keep_idx.append(j)
        else:
            evicted.append((j, p))
    orig_members, orig_grams, orig_sirens = c["members"], c["grams"], c["sirens"]
    c["members"] = [orig_members[j] for j in keep_idx]
    c["grams"] = [orig_grams[j] for j in keep_idx]
    c["sirens"] = [orig_sirens[j] for j in keep_idx]
    evicted_out = [(orig_members[j], orig_sirens[j], orig_grams[j], float(p)) for j, p in evicted]
    return evicted_out, scored_out, len(pairs)


def run_linking(blocks, reps, score, f, t0, thresholds_by_model):
    ncalls = 0
    for rep in reps:
        thr = thresholds_by_model if rep in CAL_REPS else None
        for b in blocks:
            clusters = []  # each: {"members": [text], "grams": [Counter], "sirens": [true siren]}
            for r_i, rec in enumerate(b["records"]):
                g = grams(rec["text"])
                cand = sorted(((max(cos(g, h) for h in c["grams"]), ci) for ci, c in enumerate(clusters)), reverse=True)
                cand = [ci for s, ci in cand[:3] if s >= 0.3]
                best, bestp, cscores = None, 0.0, {}
                if cand:
                    use_full_profile = rep in PROFILE_REPS
                    pairs = [((shown(clusters[ci]["members"]) if use_full_profile else [clusters[ci]["members"][0]]), rec["text"]) for ci in cand]
                    ps = score(pairs); ncalls += len(pairs)
                    cscores = dict(zip(cand, ps))
                    best, bestp = max(cscores.items(), key=lambda kv: kv[1])
                threshold_used = 0.5
                joined = False
                size_before = 0
                audits_this_record = []
                if best is not None:
                    size_before = len(clusters[best]["members"])
                    if thr is not None:
                        threshold_used = thr[bin_k(size_before)]
                    joined = bestp > threshold_used
                if joined:
                    c = clusters[best]
                    c["members"].append(rec["text"]); c["grams"].append(g); c["sirens"].append(rec["siren"]); cid = best
                    new_size = len(c["members"])
                    if rep in AUDIT_REPS and new_size in AUDIT_SIZES:
                        evicted_out, scored_out, n_audit_calls = audit_cluster(c, score, new_size)
                        ncalls += n_audit_calls
                        for ev_text, ev_siren, ev_grams, ev_p in evicted_out:
                            clusters.append({"members": [ev_text], "grams": [ev_grams], "sirens": [ev_siren]})
                            audits_this_record.append({"evicted_siren": ev_siren, "p": round(ev_p, 4), "new_cluster": len(clusters) - 1})
                        f.write(json.dumps({"rep": rep, "block": b["block"], "i": r_i, "audit": True,
                                            "trigger_cluster": cid, "milestone_size": new_size,
                                            "n_scored": n_audit_calls, "scored": scored_out,
                                            "n_evicted": len(evicted_out), "evictions": audits_this_record}) + "\n")
                else:
                    clusters.append({"members": [rec["text"]], "grams": [g], "sirens": [rec["siren"]]}); cid = len(clusters) - 1
                f.write(json.dumps({"rep": rep, "block": b["block"], "i": r_i, "siren": rec["siren"], "cluster": cid,
                                    "joined": joined, "size_before": size_before, "p_best": bestp,
                                    "threshold_used": round(threshold_used, 4),
                                    "n_cand": len(cand), "cand_scores": {str(k): round(v, 4) for k, v in cscores.items()}}) + "\n")
            f.flush()
        print("rep", rep, "done", round(time.time() - t0), "s calls", ncalls, flush=True)

    return ncalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--reps", default="anchor,profile,profile_cal,profile_audit,profile_cal_audit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=24)
    a = ap.parse_args()
    blocks = json.load(open(a.blocks))
    reps = a.reps.split(",")
    thresholds_by_model = load_thresholds(a.thresholds, a.model) if any(r in CAL_REPS for r in reps) else None
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")  # spreads across GPUs when one is too small
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
    def score(pairs):  # list of (members, query) -> list of p_yes
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
    run_linking(blocks, reps, score, f, t0, thresholds_by_model)
    f.close()
    print("FINISHED", a.model, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
