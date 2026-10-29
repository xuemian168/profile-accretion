"""Prompt-robustness replication of score_accretion.py (track: prompts).

Scores a reduced job grid (k in {1,4,16} for pos/negA/negB, plus 16
single-member judgments for negA/negB only -- see experiments/prompts/
prompts.md) under three alternative prompt wordings, keeping the next-token
YES/NO readout. Output schema matches scores_<model>.jsonl from
score_accretion.py (model, siren, q, kind, x, p_yes, mass, ntok) so the same
downstream analysis code (tables()/split() logic from scripts/build_assets.py)
applies unchanged.

python score_variants.py --items items_all.json --model Qwen/Qwen2.5-32B-Instruct \
    --variant v1 --out scores_v1_<tag>.jsonl
"""
import argparse, json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

KS = [1, 4, 16]

SYS_V0 = ("You are an entity-resolution expert for French public-procurement supplier records. "
          "Each record is 'NAME | CITY POSTCODE'. Two records refer to the same legal entity if they share the same SIREN "
          "(different establishments/branches of one company count as the SAME). Distinct legal entities in the same corporate "
          "group (subsidiaries, sister companies, franchisees) count as DIFFERENT.")

SYS_V1 = ("You work as a data-matching analyst reconciling French public-procurement supplier listings. Records are "
          "formatted as 'NAME | CITY POSTCODE'. Two records denote the same legal entity exactly when they share an "
          "identical SIREN number; subsidiaries, sister firms, and franchisees that belong to the same corporate group "
          "are DIFFERENT legal entities even when the brand name matches.")

SYS_V2 = (SYS_V0 + ' Format your answer exactly as "Answer: YES" or "Answer: NO", with no other text.')

SYS_V3 = "You are a helpful assistant."


def user_v1(members, q):
    lst = "\n".join(f"{i+1}. {m}" for i, m in enumerate(members))
    return (f"A cluster of {len(members)} confirmed record(s), all previously established as describing one single "
            f"legal entity, is listed below:\n{lst}\n\nCandidate record under review: {q}\n\nIs the candidate record "
            f"describing the same legal entity as the cluster above? Reply with a single word, YES or NO.")


def user_v2(members, q):
    rows = "\n".join(f"| {i+1} | {m} |" for i, m in enumerate(members))
    return (f"Verified profile ({len(members)} record(s) confirmed as one legal entity):\n\n"
            f"| # | Record |\n|---|---|\n{rows}\n\nRecord to classify: {q}\n\nDoes the record to classify belong to "
            f"the same legal entity as the verified profile above?")


def user_v3(members, q):
    lst = "\n".join(f"- {m}" for m in members)
    return (f"These records are known to be about the same company:\n{lst}\n\nHere is another record: {q}\n\n"
            f"Do these records refer to the same company? Answer with exactly one word: YES or NO.")


VARIANTS = {
    "v1": dict(sys=SYS_V1, user=user_v1, prefill=None),
    "v2": dict(sys=SYS_V2, user=user_v2, prefill="Answer:"),
    "v3": dict(sys=SYS_V3, user=user_v3, prefill=None),
}


def build(items):
    jobs = []
    for it in items:
        for q in ["pos", "negA", "negB"]:
            qt = it["queries"][q]["text"]
            if q in ("negA", "negB"):
                for j in range(16):
                    jobs.append(dict(siren=it["siren"], q=q, kind="single", x=j, members=[it["members"][j]], query=qt))
            for k in KS:
                jobs.append(dict(siren=it["siren"], q=q, kind="set", x=k, members=it["members"][:k], query=qt))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--out", required=True); ap.add_argument("--max_tokens_batch", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    V = VARIANTS[a.variant]
    items = json.load(open(a.items))
    jobs = build(items)
    if a.limit: jobs = jobs[:a.limit]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        print("device_map load failed, falling back:", repr(e)[:200], flush=True)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("loaded", a.model, a.variant, round(time.time() - t0), "s", flush=True)

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
        msgs = [{"role": "system", "content": V["sys"]}, {"role": "user", "content": V["user"](jb["members"], jb["query"])}]
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        if V["prefill"]:
            t = t + V["prefill"]  # literal append after the templated prompt; next token is scored right after this
        texts.append(t)
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    print("prompts", len(texts), "tokens", sum(lens), "max", max(lens), flush=True)

    out = open(a.out, "w"); done = 0; i = 0
    with torch.inference_mode():
        while i < len(order):
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
                out.write(json.dumps({"model": a.model, "variant": a.variant, "siren": jb["siren"], "q": jb["q"],
                                      "kind": jb["kind"], "x": jb["x"],
                                      "p_yes": float(py[r] / (py[r] + pn[r] + 1e-12)), "mass": float(py[r] + pn[r]),
                                      "ntok": lens[j]}) + "\n")
            i += bs; done += bs
            if done % 2000 < bs: print(done, "scored", round(time.time() - t0), "s", flush=True)
    out.close()
    print("FINISHED", a.model, a.variant, done, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
