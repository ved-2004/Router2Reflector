"""Diagnostic: effect of injected context split by whether the top-k context is answer-bearing
(discrim@k = gold option text present and no distractor present), pooled over Validation+Test.
Output: rag_eval/results/answer_bearing.csv"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SIZES  # noqa

MAP = {"wiki_dense": ("wiki_dense", 3), "wiki_dense_rr": ("wiki_dense_rr", 3), "wiki_hybrid_opt_rr": ("wiki_hybrid_opt_rr", 3),
       "arc_dense": ("arc_dense", 5), "arc_hybrid_opt_rr": ("arc_hybrid_opt_rr", 5), "R0_ft_filt": ("r0_ft_filt", 3),
       "med_dense": ("med_dense", 3), "med_dense_rr": ("med_dense_rr", 3), "med_hybrid_rr": ("med_hybrid_rr", 3),
       "med_hybrid_opt_rr": ("med_hybrid_opt_rr", 3),
       "med_bm25_rr": ("med_bm25_rr", 3), "medtb_dense_rr": ("medtb_dense_rr", 3), "medtb_hybrid_rr": ("medtb_hybrid_rr", 3)}
s1 = pd.read_csv(RES / "stage1_recall_rows.csv.gz")
out = []
for opt, (cfg, k) in MAP.items():
    a = s1[s1.config == cfg][["split", "row_id", f"recall@{k}", f"discrim@{k}"]]
    if a.empty:
        continue
    for size in SIZES:
        fr = []
        for sp in ("validation", "test"):
            f = RES / "stage2_rows" / f"{size}_{opt}_{sp}.csv.gz"
            if f.exists():
                fr.append(pd.read_csv(f).assign(split=sp))
        if not fr:
            continue
        d = pd.concat(fr).merge(a, on=["split", "row_id"])
        d = d[d.injected]
        for lab, g in (("answer-bearing (discrim)", d[d[f"discrim@{k}"] == 1]),
                       ("gold present (recall)", d[d[f"recall@{k}"] == 1]),
                       ("gold absent", d[d[f"recall@{k}"] == 0])):
            out.append({"option": opt, "size": size, "subset": lab, "n": len(g),
                        "acc_direct": round(g.direct_correct.mean() * 100, 1), "acc_ctx": round(g.final_correct.mean() * 100, 1),
                        "fixed": int(((g.final_correct == 1) & (g.direct_correct == 0)).sum()),
                        "broken": int(((g.final_correct == 0) & (g.direct_correct == 1)).sum())})
df = pd.DataFrame(out)
df.to_csv(RES / "answer_bearing.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(df)
