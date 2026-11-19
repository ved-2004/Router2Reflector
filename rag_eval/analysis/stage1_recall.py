"""Stage 1: answer-recall@k and gold-vs-distractor discriminativeness for every retrieval config.

recall@k  : normalized gold option text (>=4 chars) appears (word-bounded) in the top-k passages
discrim@k : gold appears AND none of the (>=4 char) distractors appears
Subsets: all rows, per source, and rows each model size gets wrong in the direct pass.
Output: rag_eval/results/stage1_recall.csv  (+ a pivot printed to stdout)
Run: <venv>/bin/python rag_eval/analysis/stage1_recall.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FAMILY, RES, SIZES, configs_for_corpus, load_direct, load_split, norm, r0_configs  # noqa

SPLITS = ["validation", "test"]


def main():
    cfgs = r0_configs()
    for corpus in ("wiki", "arc", "med", "medtb"):
        if (RES / "retrieval" / corpus).exists():
            cfgs.update(configs_for_corpus(corpus))
    rows = {sp: load_split(sp) for sp in SPLITS}
    wrong = {(sz, sp): set(load_direct(sz, sp).query("direct_correct == 0")["row_id"]) for sz in SIZES for sp in SPLITS}
    recs = []
    for name, per in cfgs.items():
        for sp in SPLITS:
            for r in rows[sp]:
                gold_t = norm(r["options"][r["gold"]])
                if len(gold_t) < 4:
                    continue
                dis = [norm(t) for l, t in r["options"].items() if l != r["gold"] and len(norm(t)) >= 4 and norm(t) != gold_t]
                ps = per.get(sp, {}).get(r["row_id"], [])
                rec = {"config": name, "split": sp, "row_id": r["row_id"], "source": r["source"]}
                for k in (1, 3, 5):
                    txt = " " + " ".join(norm((p.get("title") or "") + " " + p["text"]) for p in ps[:k]) + " "
                    g = f" {gold_t} " in txt
                    rec[f"recall@{k}"] = int(g)
                    rec[f"discrim@{k}"] = int(g and not any(f" {d} " in txt for d in dis))
                for sz in SIZES:
                    rec[f"wrong_{sz}"] = int(r["row_id"] in wrong[(sz, sp)])
                recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(RES / "stage1_recall_rows.csv.gz", index=False)
    out = []
    metrics = [f"{m}@{k}" for m in ("recall", "discrim") for k in (1, 3, 5)]
    for sp_name, sub0 in [("val+test", df), ("validation", df[df.split == "validation"]), ("test", df[df.split == "test"])]:
        subsets = [("all", sub0)] + [(f"src={s}", g) for s, g in sub0.groupby("source")] + \
                  [(f"wrong_{sz}", sub0[sub0[f"wrong_{sz}"] == 1]) for sz in SIZES]
        for subset, g in subsets:
            for name, gg in g.groupby("config"):
                row = {"config": name, "family": "R0" if name.startswith("r0") else FAMILY[name.split("_")[0]] +
                       ("/R2" if "_opt" in name else ""), "split": sp_name, "subset": subset, "n": len(gg)}
                row.update({m: round(gg[m].mean() * 100, 1) for m in metrics})
                out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(RES / "stage1_recall.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    p = res[(res.split == "val+test")].pivot_table(index="config", columns="subset", values="recall@3")
    print("recall@3 (val+test):\n", p[["all", "wrong_3B", "wrong_32B"] + [c for c in p.columns if c.startswith("src=")]])
    p = res[(res.split == "val+test")].pivot_table(index="config", columns="subset", values="discrim@3")
    print("\ndiscrim@3 (val+test):\n", p[["all", "wrong_3B", "wrong_32B"] + [c for c in p.columns if c.startswith("src=")]])


if __name__ == "__main__":
    main()
