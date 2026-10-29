"""Score the three re-implemented published prompting strategies with an open
chat model: ComEM-select via next-token softmax over option letters A/B/C;
BatchER-batched and LLM-CER-cluster via greedy generation + regex parsing.

python score_systems.py --jobs_dir data/raw/systems --model Qwen/Qwen2.5-32B-Instruct \
    --out_dir data/raw/systems --tag Qwen2.5-32B-Instruct
"""
import argparse, json, os, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_jobs(path):
    return [json.loads(l) for l in open(path)]


def apply_template(tok, msgs, kw):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


# --------------------------------------------------------- next-token (ComEM)
def run_comem(jobs, tok, model, kw, out_path, max_tokens_batch):
    def ids_for(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tok.encode(v, add_special_tokens=False)
                if t:
                    s.add(t[0])
        return sorted(s)
    IDS = {L: ids_for([L, L.lower()]) for L in ["A", "B", "C"]}
    texts = [apply_template(tok, j["messages"], kw) for j in jobs]
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    out = open(out_path, "w")
    t0 = time.time(); i = 0; done = 0
    with torch.inference_mode():
        while i < len(order):
            bs = 1
            while i + bs < len(order) and (bs + 1) * lens[order[i + bs]] <= max_tokens_batch and bs < 64:
                bs += 1
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            try:
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(**enc).logits[:, -1, :].float()
            pr = torch.softmax(logits, dim=-1)
            pA = pr[:, IDS["A"]].sum(-1); pB = pr[:, IDS["B"]].sum(-1); pC = pr[:, IDS["C"]].sum(-1)
            for r, j in enumerate(idx):
                jb = jobs[j]
                out.write(json.dumps({**{k: jb[k] for k in ("siren", "qtype", "label", "k", "strategy")},
                                       "p_A": float(pA[r]), "p_B": float(pB[r]), "p_C": float(pC[r]),
                                       "mass": float(pA[r] + pB[r] + pC[r])}) + "\n")
            i += bs; done += bs
            if done % 1000 < bs:
                print("comem", done, "/", len(jobs), round(time.time() - t0), "s", flush=True)
    out.close()


# ---------------------------------------------------------- generation (BatchER, LLM-CER)
def run_generate(jobs, tok, model, kw, out_path, max_new_tokens_fn, batch_size=8):
    texts = [apply_template(tok, j["messages"], kw) for j in jobs]
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])
    out = open(out_path, "w")
    t0 = time.time(); done = 0
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            mnt = max(max_new_tokens_fn(jobs[j]) for j in idx)
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False, temperature=None, top_p=None,
                                  pad_token_id=tok.pad_token_id)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            texts_out = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for r, j in enumerate(idx):
                jb = jobs[j]
                out.write(json.dumps({**{k: jb[k] for k in jb if k != "messages"}, "completion": texts_out[r]}) + "\n")
            done += len(idx)
            if done % 200 < len(idx):
                print("generate", done, "/", len(jobs), round(time.time() - t0), "s", flush=True)
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max_tokens_batch", type=int, default=16000)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        print("device_map load failed, falling back:", repr(e)[:200], flush=True)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("loaded", a.model, round(time.time() - t0), "s", flush=True)
    kw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}

    comem = load_jobs(os.path.join(a.jobs_dir, "jobs_comem_select.jsonl"))
    batcher = load_jobs(os.path.join(a.jobs_dir, "jobs_batcher.jsonl"))
    llmcer = load_jobs(os.path.join(a.jobs_dir, "jobs_llmcer_cluster.jsonl"))
    if a.limit:
        comem, batcher, llmcer = comem[:a.limit], batcher[:a.limit], llmcer[:a.limit]

    run_comem(comem, tok, model, kw, os.path.join(a.out_dir, f"comem_select_{a.tag}.jsonl"), a.max_tokens_batch)
    print("FINISHED comem_select", a.tag, round(time.time() - t0), "s", flush=True)

    run_generate(batcher, tok, model, kw, os.path.join(a.out_dir, f"batcher_{a.tag}.jsonl"),
                 max_new_tokens_fn=lambda j: 8 * j["k"] + 24, batch_size=a.gen_batch_size)
    print("FINISHED batcher", a.tag, round(time.time() - t0), "s", flush=True)

    run_generate(llmcer, tok, model, kw, os.path.join(a.out_dir, f"llmcer_cluster_{a.tag}.jsonl"),
                 max_new_tokens_fn=lambda j: 10 * j["n_recs"] + 40, batch_size=a.gen_batch_size)
    print("FINISHED llmcer_cluster", a.tag, round(time.time() - t0), "s", flush=True)
    print("ALL DONE", a.tag, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
