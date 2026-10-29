"""Accretion-bias experiment: next-token YES/NO probability from open-weight chat models.

python score_accretion.py --items items_all.json --model Qwen/Qwen2.5-32B-Instruct --out scores_<tag>.jsonl
"""
import argparse, json, time, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYS = ("You are an entity-resolution expert for French public-procurement supplier records. "
       "Each record is 'NAME | CITY POSTCODE'. Two records refer to the same legal entity if they share the same SIREN "
       "(different establishments/branches of one company count as the SAME). Distinct legal entities in the same corporate "
       "group (subsidiaries, sister companies, franchisees) count as DIFFERENT.")
KS = [1, 2, 4, 8, 16]


def user_prompt(members, q, extra=""):
    lst = "\n".join(f"{i+1}. {m}" for i, m in enumerate(members))
    ex = (extra + " ") if extra else ""
    return (f"Known entity — the following {len(members)} record(s) were verified to belong to ONE legal entity:\n{lst}\n\n"
            f"New record: {q}\n\nDoes the new record belong to that same legal entity? {ex}Answer with exactly one word: YES or NO.")


def build(items):
    jobs = []
    for it in items:
        for q in ["pos", "negA", "negB"]:
            qt = it["queries"][q]["text"]
            for j in range(16):
                jobs.append(dict(siren=it["siren"], q=q, kind="single", x=j, members=[it["members"][j]], query=qt))
            for k in KS[1:]:
                jobs.append(dict(siren=it["siren"], q=q, kind="set", x=k, members=it["members"][:k], query=qt))
            jobs.append(dict(siren=it["siren"], q=q, kind="set_rev", x=16, members=it["members"][:16][::-1], query=qt))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max_tokens_batch", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.items)]
    if a.limit: jobs = jobs[:a.limit]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")  # spreads across GPUs when one is too small
    except Exception as e:
        print("device_map load failed, falling back:", repr(e)[:200], flush=True)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("loaded", a.model, round(time.time() - t0), "s", flush=True)

    def ids_for(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t: s.add(t[0])
        return sorted(s)
    YES = ids_for(["YES", "Yes", "yes"]); NO = ids_for(["NO", "No", "no"])
    print("yes ids", YES, "no ids", NO, flush=True)

    kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}
    texts = []
    for jb in jobs:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user_prompt(jb["members"], jb["query"], jb.get("extra", ""))}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    print("prompts", len(texts), "tokens", sum(lens), "max", max(lens), flush=True)

    out = open(a.out, "w"); done = 0; i = 0
    with torch.inference_mode():
        while i < len(order):
            L = lens[order[min(i + 1, len(order) - 1)]]
            bs = 1
            while i + bs < len(order) and (bs + 1) * lens[order[i + bs]] <= a.max_tokens_batch and bs < 64:
                bs += 1
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            try:
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(**enc).logits[:, -1, :].float()
            pr = torch.softmax(logits, dim=-1)
            py = pr[:, YES].sum(-1); pn = pr[:, NO].sum(-1)
            for r, j in enumerate(idx):
                jb = jobs[j]
                out.write(json.dumps({"model": a.model, "siren": jb["siren"], "q": jb["q"], "kind": jb["kind"], "x": jb["x"],
                                      "p_yes": float(py[r] / (py[r] + pn[r] + 1e-12)), "mass": float(py[r] + pn[r]),
                                      "ntok": lens[j]}) + "\n")
            i += bs; done += bs
            if done % 2000 < bs: print(done, "scored", round(time.time() - t0), "s", flush=True)
    out.close()
    print("FINISHED", a.model, done, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
