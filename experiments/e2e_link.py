"""End-to-end online linking on confusable blocks with an LLM matcher.

python e2e_link.py --blocks blocks_e2e.json --model Qwen/Qwen2.5-32B-Instruct --reps profile,anchor,pairmax --out e2e_<tag>.jsonl
Each record arrives in stream order; candidates = existing clusters whose best char-3gram cosine to the
record name is >= 0.3 (top 3). The record joins the best-scoring candidate if P(YES) > 0.5, else opens a new cluster.
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


def shown(members):  # founding member + most recent ones, at most MAXM
    return members if len(members) <= MAXM else [members[0]] + members[-(MAXM - 1):]


def run_linking(blocks, reps, score, f, t0):
    ncalls = 0
    for rep in reps:
        for b in blocks:
            clusters = []  # each: {"members": [text], "grams": [Counter], "sirens": [true siren]}
            for r_i, rec in enumerate(b["records"]):
                g = grams(rec["text"])
                cand = sorted(((max(cos(g, h) for h in c["grams"]), ci) for ci, c in enumerate(clusters)), reverse=True)
                cand = [ci for s, ci in cand[:3] if s >= 0.3]
                best, bestp, cscores = None, 0.0, {}
                if cand:
                    if rep == "pairmax":
                        pairs, owner = [], []
                        for ci in cand:
                            for m in shown(clusters[ci]["members"]):
                                pairs.append(([m], rec["text"])); owner.append(ci)
                        ps = score(pairs); ncalls += len(pairs)
                        for ci, p in zip(owner, ps): cscores[ci] = max(cscores.get(ci, 0.0), p)
                    else:
                        pairs = [((shown(clusters[ci]["members"]) if rep == "profile" else [clusters[ci]["members"][0]]), rec["text"]) for ci in cand]
                        ps = score(pairs); ncalls += len(pairs)
                        cscores = dict(zip(cand, ps))
                    best, bestp = max(cscores.items(), key=lambda kv: kv[1])
                if best is not None and bestp > 0.5:
                    c = clusters[best]; size_before = len(c["members"])
                    c["members"].append(rec["text"]); c["grams"].append(g); c["sirens"].append(rec["siren"]); cid = best
                else:
                    clusters.append({"members": [rec["text"]], "grams": [g], "sirens": [rec["siren"]]}); cid = len(clusters) - 1; size_before = 0
                f.write(json.dumps({"rep": rep, "block": b["block"], "i": r_i, "siren": rec["siren"], "cluster": cid,
                                    "joined": size_before > 0, "size_before": size_before, "p_best": bestp,
                                    "n_cand": len(cand), "cand_scores": {str(k): round(v, 4) for k, v in cscores.items()}}) + "\n")
            f.flush()
        print("rep", rep, "done", round(time.time() - t0), "s calls", ncalls, flush=True)

    return ncalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--reps", default="profile,anchor,pairmax"); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=24)
    a = ap.parse_args()
    blocks = json.load(open(a.blocks))
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
    run_linking(blocks, a.reps.split(","), score, f, t0)
    f.close()
    print("FINISHED", a.model, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
