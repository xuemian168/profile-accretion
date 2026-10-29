"""Closed-model (Claude) track: build the 100-entity subset (k in {1,4,16} x {pos,negA,negB})
plus the k=4 clean/contaminated snowball pairs, then sample each prompt 3x.

Sampling goes through `host.llm`, an in-house batching client for the Anthropic Messages API:
it takes a list of request dicts (model, system, messages, max_tokens) and returns the
completions with their `usage` fields. The script was run interactively with `host` bound in
the session; to rerun it, bind any client with the same interface.

Notes from the run (see data/derived/closed/FINDINGS.md):
  - The sampling environment capped total usage at about 2.0M tokens across models. The first
    Sonnet run reached that cap after 2,610 of 3,336 queued (prompt, sample) requests; the
    remaining Sonnet prompts and both Haiku 5.5 runs were completed in later shards.
  - Each call costs about 5,000-5,500 tokens, dominated by the cached system prompt; the YES/NO
    answer itself uses at most 20 output tokens.
"""
import json, re, random, time

SYS = ("You are an entity-resolution expert for French public-procurement supplier records. "
       "Each record is 'NAME | CITY POSTCODE'. Two records refer to the same legal entity if they share the same SIREN "
       "(different establishments/branches of one company count as the SAME). Distinct legal entities in the same corporate "
       "group (subsidiaries, sister companies, franchisees) count as DIFFERENT.")


def user_prompt(members, q):
    lst = "\n".join(f"{i+1}. {m}" for i, m in enumerate(members))
    return (f"Known entity — the following {len(members)} record(s) were verified to belong to ONE legal entity:\n{lst}\n\n"
            f"New record: {q}\n\nDoes the new record belong to that same legal entity? Answer with exactly one word: YES or NO.")


def parse_yesno(text):
    if not text:
        return None
    m = re.search(r"\b(YES|NO)\b", text.upper())
    return None if not m else (1 if m.group(1) == "YES" else 0)


def build_prompts(items_all_path, jobs_mitig_path, seed=20261027, n_entities=100):
    items = json.load(open(items_all_path))
    rng = random.Random(seed)
    sel_sirens = sorted(rng.sample([it["siren"] for it in items], n_entities))
    by_siren = {it["siren"]: it for it in items}
    sel_items = [by_siren[s] for s in sel_sirens]

    KS = [1, 4, 16]
    profile = []
    for it in sel_items:
        for qk in ["pos", "negA", "negB"]:
            qt = it["queries"][qk]["text"]
            for k in KS:
                members = it["members"][:k]
                profile.append(dict(pid=f"profile|{it['siren']}|{qk}|k{k}", siren=it["siren"], q=qk, kind="set",
                                     k=k, members=members, query=qt, prompt=user_prompt(members, qt)))
    assert len(profile) == n_entities * 3 * 3

    jobs_mitig = [json.loads(l) for l in open(jobs_mitig_path)]
    sel_set = set(sel_sirens)
    snow_raw = [r for r in jobs_mitig if r["kind"] in ("clean", "contaminated") and r["x"] == 4 and r["siren"] in sel_set]
    snow = [dict(pid=f"snow|{r['siren']}|{r['q']}|{r['kind']}|x{r['x']}", siren=r["siren"], q=r["q"], kind=r["kind"],
                 k=r["x"], members=r["members"], query=r["query"], prompt=user_prompt(r["members"], r["query"]))
            for r in snow_raw]
    return sel_sirens, profile + snow


def sample_model(all_prompts, model, out_path, n_samples=3, chunk=30, conc=10, max_tokens=20,
                  order_seed=777, resume_from=0, stop_token_budget=None, wallclock_cap_s=None):
    """Sample `all_prompts` n_samples times each via host.llm, writing incrementally.
    Stops cleanly (recording stop_reason) on: a budget RuntimeError from the client, a chunk
    with a majority of errors, or the token/wallclock caps. `host` must be bound in the
    calling namespace (see the module docstring)."""
    order = list(range(len(all_prompts)))
    random.Random(order_seed).shuffle(order)
    queue = [(pidx, sidx) for pidx in order for sidx in range(n_samples)]
    mode = "a" if resume_from else "w"
    fout = open(out_path, mode)
    t0 = time.time(); cum_tokens = 0; n_done = 0; n_err = 0; stop_reason = None
    i = resume_from
    while i < len(queue):
        ch = queue[i:i + chunk]
        reqs = [{"prompt": all_prompts[p]["prompt"], "system": SYS, "max_tokens": max_tokens, "model": model} for p, _ in ch]
        try:
            outs = host.llm(reqs, max_concurrency=conc)
        except Exception as e:
            stop_reason = f"exception: {repr(e)[:400]}"; break
        chunk_err = 0
        for (pidx, sidx), o in zip(ch, outs):
            p = all_prompts[pidx]; err = o.get("error"); chunk_err += bool(err)
            u = o.get("usage", {}) or {}
            cum_tokens += sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"))
            fout.write(json.dumps(dict(pid=p["pid"], siren=p["siren"], q=p["q"], kind=p["kind"], k=p["k"], sample=sidx,
                                        model=o.get("model"), text=o.get("text"), ans=parse_yesno(o.get("text")),
                                        err=err[:300] if err else None)) + "\n")
        fout.flush(); n_done += len(ch) - chunk_err; n_err += chunk_err; i += len(ch)
        if chunk_err > len(ch) // 2:
            stop_reason = f"majority errors in chunk at i={i}"; break
        if stop_token_budget and cum_tokens > stop_token_budget:
            stop_reason = f"hit token budget at i={i}"; break
        if wallclock_cap_s and time.time() - t0 > wallclock_cap_s:
            stop_reason = f"hit wallclock cap at i={i}"; break
    fout.close()
    return dict(n_done=n_done, n_err=n_err, i_reached=i, queue_len=len(queue), cum_tokens=cum_tokens, stop_reason=stop_reason)
