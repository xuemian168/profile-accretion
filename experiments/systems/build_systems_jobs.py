"""Build job files for the three re-implemented published prompting strategies
(BatchER batched-pairwise, ComEM selecting, LLM-CER in-context clustering) on
the same controlled FOPPA accretion design (k in {1,4,16}, fixed query).

python build_systems_jobs.py --items ../../data/raw/items_all.json --out_dir ../../data/raw/systems
"""
import argparse, json, os

KS = [1, 4, 16]
QTYPES = ["pos", "negA", "negB"]

SYS_MATCH = ("You are an entity-resolution expert for French public-procurement supplier records. "
             "Each record is 'NAME | CITY POSTCODE'. Two records refer to the same legal entity if they share the same SIREN "
             "(different establishments/branches of one company count as the SAME). Distinct legal entities in the same corporate "
             "group (subsidiaries, sister companies, franchisees) count as DIFFERENT.")

# ---------------------------------------------------------------- BatchER ---
FIXED_DEMO_MATCH = ("Record A: SOCIETE GENERALE | PARIS 75009\nRecord B: SOCIETE GENERALE AGENCE LYON | LYON 69002\n"
                     "Same legal entity? YES/NO\nAnswer: YES")
FIXED_DEMO_NONMATCH = ("Record A: CARREFOUR | PARIS 75001\nRecord B: CARREFOUR MARKET FRANCHISE SUD | TOULOUSE 31000\n"
                        "Same legal entity? YES/NO\nAnswer: NO")


def batcher_prompt(members, query):
    demos = FIXED_DEMO_MATCH + "\n\n" + FIXED_DEMO_NONMATCH
    qs = "\n\n".join(
        f"Q{i+1}.\nRecord A: {m}\nRecord B: {query}\nSame legal entity? YES/NO"
        for i, m in enumerate(members)
    )
    user = (f"Here are {len(members)} examples of the task, then {len(members)} questions about new record pairs.\n\n"
            f"--- Examples ---\n{demos}\n\n--- Questions ---\n{qs}\n\n"
            f"Answer every question in order, one per line, exactly as 'Qi: YES' or 'Qi: NO' (no explanations).")
    return [{"role": "system", "content": SYS_MATCH}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ ComEM ---
def comem_select_prompt(members_a, members_b, query):
    lst_a = "\n".join(f"  - {m}" for m in members_a)
    lst_b = "\n".join(f"  - {m}" for m in members_b)
    user = (f"Candidate cluster A ({len(members_a)} verified record(s) of one legal entity):\n{lst_a}\n\n"
            f"Candidate cluster B ({len(members_b)} verified record(s) of a different legal entity):\n{lst_b}\n\n"
            f"Query record: {query}\n\n"
            "Which cluster does the query record belong to? Answer with exactly one letter:\n"
            "A = belongs to candidate cluster A\nB = belongs to candidate cluster B\n"
            "C = belongs to neither (none of the above)\nAnswer:")
    return [{"role": "system", "content": SYS_MATCH}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- LLM-CER ---
def llmcer_cluster_prompt(members, query, distractors):
    recs = list(members) + [query] + list(distractors)
    lst = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recs))
    n_query_idx = len(members) + 1
    user = (f"Here are {len(recs)} supplier records, numbered 1-{len(recs)}:\n{lst}\n\n"
            "Group these records into clusters such that records in the same cluster refer to the SAME legal entity "
            "(same SIREN), and records in different clusters refer to different legal entities. "
            "Return ONLY a Python-style list of clusters, each cluster a list of record numbers, e.g. "
            "[[1,2,3],[4,5]]. Every record number 1.." + str(len(recs)) + " must appear exactly once.")
    return [{"role": "system", "content": SYS_MATCH}, {"role": "user", "content": user}], n_query_idx, len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    items = json.load(open(a.items))
    n = len(items)
    os.makedirs(a.out_dir, exist_ok=True)

    batcher_jobs, comem_jobs, llmcer_jobs = [], [], []
    for i, it in enumerate(items):
        siren = it["siren"]
        members16 = it["members"]
        distractor_entity = items[(i + 1) % n]
        for qtype in QTYPES:
            q = it["queries"][qtype]["text"]
            label = it["queries"][qtype]["label"]  # 1 for pos, 0 for negA/negB
            for k in KS:
                members_k = members16[:k]
                # BatchER
                msgs = batcher_prompt(members_k, q)
                batcher_jobs.append({"siren": siren, "qtype": qtype, "label": label, "k": k,
                                      "strategy": "batcher", "messages": msgs})
                # ComEM select
                distractor_k = distractor_entity["members"][:k]
                msgs2 = comem_select_prompt(members_k, distractor_k, q)
                comem_jobs.append({"siren": siren, "qtype": qtype, "label": label, "k": k,
                                    "strategy": "comem_select", "messages": msgs2,
                                    "distractor_siren": distractor_entity["siren"]})
                # LLM-CER cluster
                distractors2 = distractor_entity["members"][:2]
                msgs3, query_idx, n_recs = llmcer_cluster_prompt(members_k, q, distractors2)
                llmcer_jobs.append({"siren": siren, "qtype": qtype, "label": label, "k": k,
                                     "strategy": "llmcer_cluster", "messages": msgs3,
                                     "query_idx": query_idx, "n_profile": k, "n_recs": n_recs,
                                     "distractor_siren": distractor_entity["siren"]})

    for name, jobs in [("batcher", batcher_jobs), ("comem_select", comem_jobs), ("llmcer_cluster", llmcer_jobs)]:
        path = os.path.join(a.out_dir, f"jobs_{name}.jsonl")
        with open(path, "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")
        print(name, len(jobs), "->", path)


if __name__ == "__main__":
    main()
