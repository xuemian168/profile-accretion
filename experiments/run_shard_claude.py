"""Accretion-bias pilot: run one shard of LLM match judgments.

Usage in an interactive session where `host` is bound (a batching client for the Anthropic
Messages API; see experiments/closed/build_and_sample_closed.py):
    ITEMS_PATH = "<path to items.json>"; START, END = 0, 25; MODEL = None; MAX_TOKENS = 5; OUT = "res_shard.jsonl"
    exec(open("run_shard.py").read())
"""
import json, time

SYS = ("You are an entity-resolution expert for French public-procurement supplier records. "
       "Each record is 'NAME | CITY POSTCODE'. Two records refer to the same legal entity if they share the same SIREN "
       "(different establishments/branches of one company count as the SAME). Distinct legal entities in the same corporate "
       "group (subsidiaries, sister companies, franchisees) count as DIFFERENT.")
KS = [1, 2, 4, 8, 16]


def prompt(members, q):
    lst = "\n".join(f"{i+1}. {m}" for i, m in enumerate(members))
    return (f"Known entity — the following {len(members)} record(s) were verified to belong to ONE legal entity:\n{lst}\n\n"
            f"New record: {q}\n\nDoes the new record belong to that same legal entity? Think briefly (at most two sentences), "
            f"then end with a final line that is exactly 'ANSWER: YES' or 'ANSWER: NO'. If uncertain, give your best guess.")


def parse(t):
    import re
    t = (t or "").upper()
    m = re.findall(r"ANSWER:\s*(YES|NO)", t)
    if not m:
        m = re.findall(r"\b(YES|NO)\b", t)
    return None if not m else (1 if m[-1] == "YES" else 0)


def build(items_sub):
    jobs = []
    for it in items_sub:
        for q in ["pos", "negA", "negB"]:
            qt = it["queries"][q]["text"]
            if q != "pos":  # singles (for the OR-of-singles null) only for negatives; pos gets k=1 single only
                for j in range(16):
                    jobs.append(("single", it["siren"], q, j, prompt([it["members"][j]], qt)))
            else:
                jobs.append(("single", it["siren"], q, 0, prompt([it["members"][0]], qt)))
            for k in KS[1:]:
                jobs.append(("set", it["siren"], q, k, prompt(it["members"][:k], qt)))
    return jobs


def run(jobs, model=None, max_tokens=5, conc=16, path="res.jsonl", chunk_n=400):
    t0 = time.time(); n_ok = n_err = 0
    with open(path, "w") as f:
        for c in range(0, len(jobs), chunk_n):
            chunk = jobs[c:c + chunk_n]
            reqs = [{"prompt": p, "system": SYS, "max_tokens": max_tokens, "temperature": 0,
                     **({"model": model} if model else {})} for *_, p in chunk]
            outs = host.llm(reqs, max_concurrency=conc)
            e = 0
            for (kind, s, q, x, _), o in zip(chunk, outs):
                err = o.get("error")
                e += bool(err)
                f.write(json.dumps({"kind": kind, "siren": s, "q": q, "x": x, "ans": parse(o.get("text")),
                                    "raw": (o.get("text") or "")[-200:], "model": o.get("model"),
                                    "err": err[:160] if err else None}) + "\n")
            f.flush(); n_err += e; n_ok += len(chunk) - e
            print(c + len(chunk), "done", round(time.time() - t0), "s errors", e, flush=True)
            if e > len(chunk) // 2:
                print("ABORT:", next(o.get("error") for o in outs if o.get("error"))[:200]); break
    return n_ok, n_err


items_all = json.load(open(ITEMS_PATH))
shard = items_all[START:END]
jobs = build(shard)
print("shard", START, END, "entities", len(shard), "jobs", len(jobs), "model", MODEL)
n_ok, n_err = run(jobs, model=MODEL, max_tokens=MAX_TOKENS, path=OUT)
print("FINAL ok", n_ok, "err", n_err)
