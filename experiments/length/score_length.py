"""Length control for profile accretion: does a profile of k copies of the founding record
(same content, longer prompt, larger stated record count) shift the decision like k distinct members?

Jobs per entity and query (pos, negA, negB):
  set   x=1   founding member only
  set   x=16  first 16 verified members (as in the main design)
  rep   x=4   founding member repeated 4 times
  rep   x=16  founding member repeated 16 times
Same system prompt, user prompt, model loading and next-token P(YES) readout as score_accretion.py.

python score_length.py --items items_all.json --model <hf id> --out length_<tag>.jsonl
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import score_accretion as sa


def build(items):
    jobs = []
    for it in items:
        f = it["members"][0]
        for q in ["pos", "negA", "negB"]:
            qt = it["queries"][q]["text"]
            jobs.append(dict(siren=it["siren"], q=q, kind="set", x=1, members=[f], query=qt))
            jobs.append(dict(siren=it["siren"], q=q, kind="set", x=16, members=it["members"][:16], query=qt))
            for k in (4, 16):
                jobs.append(dict(siren=it["siren"], q=q, kind="rep", x=k, members=[f] * k, query=qt))
    return jobs


sa.build = build
if __name__ == "__main__":
    sa.main()
