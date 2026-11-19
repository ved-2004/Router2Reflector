"""Soft contamination check on RETRIEVED text: fraction of the question's word 8-grams that occur in the
top-5 passages of each config (near-verbatim copies of the question would score ~1.0).
Complements the exact-substring full-corpus scan (contam_all on Modal -> results/contam.json).
Output: rag_eval/results/contam_retrieved.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, configs_for_corpus, load_split, norm  # noqa


def grams(t, n=8):
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    recs = []
    for corpus in ("wiki", "arc", "med", "medtb"):
        if not (RES / "retrieval" / corpus).exists():
            continue
        cfgs = configs_for_corpus(corpus)
        for name in (f"{corpus}_dense", f"{corpus}_bm25", f"{corpus}_hybrid_rr", f"{corpus}_hybrid_opt_rr"):
            for sp in ("validation", "test"):
                for r in load_split(sp):
                    g = grams(norm(r["question"]))
                    if len(g) < 3:
                        continue
                    txt = " ".join(norm(p["text"]) for p in cfgs[name][sp].get(r["row_id"], [])[:5])
                    ov = sum(x in txt for x in g) / len(g)
                    recs.append({"config": name, "split": sp, "row_id": r["row_id"], "source": r["source"], "overlap": ov})
    df = pd.DataFrame(recs)
    df.to_csv(RES / "contam_retrieved.csv", index=False)
    s = df.groupby(["config", "source"]).overlap.agg(mean="mean", n_ge50=lambda x: int((x >= 0.5).sum()),
                                                      n_ge90=lambda x: int((x >= 0.9).sum()), n="size")
    pd.set_option("display.max_rows", 200)
    print(s.round(3))


if __name__ == "__main__":
    main()
