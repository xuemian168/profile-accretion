"""BatchER-style format: does the answer to the SAME (founding member, query)
sub-question change when it is batched with 15 further members (k=16) instead of
asked alone (k=1)?  Separates within-prompt context effects from the extra chances
added by OR aggregation.  Writes data/derived/systems/batcher_q1_shift.csv."""
import json, re, glob, os
import pandas as pd
from scipy.stats import binomtest
R = os.path.join(os.path.dirname(__file__), '..', '..')
rows = []
for f in sorted(glob.glob(os.path.join(R, 'data/raw/systems/batcher_*.jsonl'))):
    m = os.path.basename(f)[len('batcher_'):-len('.jsonl')]
    for l in open(f):
        d = json.loads(l)
        a = {int(i): v for i, v in re.findall(r'Q(\d+)\s*:\s*(YES|NO)', d['completion'].upper())}
        ys = [a.get(i) == 'YES' for i in range(1, d['k'] + 1)]
        rows.append(dict(model=m, siren=d['siren'], qtype=d['qtype'], k=d['k'],
                         q1=(a.get(1) == 'YES') if 1 in a else None,
                         frac_yes=sum(ys) / d['k'] if len(a) >= d['k'] else None))
df = pd.DataFrame(rows).drop_duplicates(['model', 'siren', 'qtype', 'k'])
neg = df[df.qtype != 'pos']
w = neg.dropna(subset=['q1']).pivot_table(index=['model', 'siren', 'qtype'], columns='k', values='q1').dropna()
fr = neg.dropna(subset=['frac_yes']).groupby(['model', 'k']).frac_yes.mean().unstack()
out = []
for m, g in w.groupby(level=0):
    up = int(((g[1] == 0) & (g[16] == 1)).sum()); dn = int(((g[1] == 1) & (g[16] == 0)).sum())
    out.append(dict(model=m, n=len(g), q1_yes_k1=100 * g[1].mean(), q1_yes_k16=100 * g[16].mean(),
                    up=up, down=dn, mcnemar_p=binomtest(up, up + dn).pvalue if up + dn else 1.0,
                    mean_subq_yes_k1=100 * fr.loc[m, 1], mean_subq_yes_k16=100 * fr.loc[m, 16]))
pd.DataFrame(out).to_csv(os.path.join(R, 'data/derived/systems/batcher_q1_shift.csv'), index=False)
print(pd.DataFrame(out).round(3).to_string())
