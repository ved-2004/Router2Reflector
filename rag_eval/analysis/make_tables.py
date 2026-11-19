"""Emit markdown tables for REPORT.md from results/stage1_recall.csv and results/stage2_accuracy.csv.
Also evaluates a per-source router over several options ("combo": for each source pick, on Validation,
the option -- or none -- with the best Validation accuracy for that size; apply to Test).
Run: <venv>/bin/python rag_eval/analysis/make_tables.py > rag_eval/results/tables.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SIZES  # noqa

LABEL = {
    "R0_ft_filt": "R0 BM25 wiki leads (current, filtered)",
    "wiki_dense": "R1 wiki full-article, dense bge",
    "wiki_dense_rr": "R1 wiki dense + bge-reranker",
    "wiki_dense__L3000": "R1 wiki dense, 5 psg / 3000 chars",
    "wiki_hybrid_opt_rr": "R2 wiki option-aware hybrid + reranker",
    "med_dense": "R3 MedCorp (tb+SP+PubMed) MedCPT",
    "med_dense_rr": "R3 MedCorp MedCPT + MedCPT-CE",
    "med_hybrid_rr": "R3 MedCorp BM25+MedCPT + MedCPT-CE",
    "med_hybrid_opt_rr": "R2+R3 MedCorp option-aware + MedCPT-CE",
    "med_bm25_rr": "R3 MedCorp BM25 + MedCPT-CE",
    "med_hybrid_rr__L3000": "R3 MedCorp hybrid + CE, 5 psg / 3000 chars",
    "medtb_dense_rr": "R3 textbooks+StatPearls MedCPT + CE",
    "medtb_hybrid_rr": "R3 textbooks+StatPearls BM25+MedCPT + CE",
    "arc_dense": "R4 ARC corpus dense bge (5 sent.)",
    "arc_hybrid_opt_rr": "R4 ARC option-aware hybrid + reranker",
    "gen32B": "R5 generate-then-read (32B writes)",
    "genself": "R5 self-generated (each size writes)",
    "gen3B": "R5 3B-generated", "gen7B": "R5 7B-generated", "gen14B": "R5 14B-generated",
    "oracle_sciq": "R6 oracle SciQ support passage",
}
SELF = {"3B": "gen3B", "7B": "gen7B", "14B": "gen14B", "32B": "gen32B"}


def cell(r):
    return f"{r.delta_pp:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]"


def main():
    a = pd.read_csv(RES / "stage2_accuracy.csv")
    # synthesize the "genself" row: each size with its own generator
    gs = []
    for sz, opt in SELF.items():
        gs.append(a[(a["size"] == sz) & (a.option == opt)].assign(option="genself"))
    a = pd.concat([a] + gs)
    for gate in ("always", "route", "H", "nli", "src"):
        t = a[(a.split == "test") & (a.gate == gate)]
        if t.empty:
            continue
        piv = t.assign(c=t.apply(cell, axis=1)).pivot(index="option", columns="size", values="c")[SIZES]
        mean = t.groupby("option").delta_pp.mean().rename("mean")
        piv = piv.join(mean).sort_values("mean", ascending=False)
        inj = t.groupby("option").inject_rate.mean().round(0)
        print(f"\n#### Test delta accuracy (pp) vs direct, gate = `{gate}`  [paired-bootstrap 95% CI]\n")
        print("| option | " + " | ".join(SIZES) + " | mean | inject % |")
        print("|---|" + "---|" * (len(SIZES) + 2))
        for opt, r in piv.iterrows():
            if opt.startswith("gen") and opt not in ("gen32B",):
                if opt != "genself":
                    continue
            print(f"| {LABEL.get(opt, opt)} | " + " | ".join(r[s] for s in SIZES) + f" | {r['mean']:+.1f} | {inj[opt]:.0f} |")
    # fixed/broken (always) on test
    t = a[(a.split == "test") & (a.gate == "always")]
    print("\n#### Fixed / broken counts (Test, always-on)\n")
    print("| option | " + " | ".join(SIZES) + " |")
    print("|---|" + "---|" * len(SIZES))
    for opt, g in t.groupby("option"):
        if opt.startswith("gen") and opt not in ("gen32B", "genself"):
            continue
        g = g.set_index("size")
        print(f"| {LABEL.get(opt, opt)} | " + " | ".join(f"+{g.loc[s, 'fixed']}/-{g.loc[s, 'broken']}" for s in SIZES) + " |")
    # per-source
    s = pd.read_csv(RES / "stage2_per_source.csv")
    s = s[(s.split == "test") & (s.gate == "always")]
    print("\n#### Per-source Test delta (pp, always-on; n=50 per source, 1 question = 2 pp)\n")
    srcs = sorted(s.source.unique())
    print("| option | size | " + " | ".join(srcs) + " |")
    print("|---|---|" + "---|" * len(srcs))
    for opt in [o for o in LABEL if o in set(s.option)]:
        if opt.startswith("gen") and opt != "gen32B":
            continue
        for sz in SIZES:
            g = s[(s.option == opt) & (s["size"] == sz)].set_index("source")
            if g.empty:
                continue
            print(f"| {LABEL[opt]} | {sz} | " + " | ".join(f"{g.loc[x, 'delta_pp']:+.0f}" for x in srcs) + " |")


if __name__ == "__main__":
    main()
