"""
Fetch Wikipedia passages locally (Wikipedia throttles Modal's shared datacenter IPs heavily:
~14 s/question there vs ~1 s locally). Produces the same retrieval_v2 JSON as
``r2r_modal.retrieve`` but with ``sim = null``; similarities are filled in on Modal by
``--stage score`` after uploading:

  python modal_app/local_retrieve.py --split validation --split test
  modal volume put -f r2r-results outputs/retrieval_v2/test.json retrieval_v2/test.json
  modal run modal_app/r2r_modal.py --stage score --splits test
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rag_pipeline import RAGRetriever  # noqa: E402
from reflector_core import get_passage_for_row  # noqa: E402

SPLIT_FILES = {"train": "Train.csv", "validation": "Validation.csv", "test": "Test.csv"}
MODES = {"": "opensearch", "_ft": "fulltext"}   # same suffixes as r2r_modal.RETRIEVAL_MODES


def run(split: str, out_dir: Path) -> None:
    out = out_dir / f"{split}.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    df = pd.read_csv(REPO / "data" / "raw" / SPLIT_FILES[split]).reset_index(drop=True)
    rets = {suf: RAGRetriever(top_k=3, search_mode=m) for suf, m in MODES.items()}
    t0, n_new = time.time(), 0
    for i, r in df.iterrows():
        key = str(i)
        if key in done and not any(done[key].get(f"failed{s}") for s in MODES):
            continue
        passage = get_passage_for_row(str(r["source"]), r.get("context"))
        q = str(r["question"])
        full_q = q if not passage else f"{passage}\n\n{q}"      # same as r2r_baseline_comparison.py
        rec = {"queries": rets[""].extract_queries(full_q)}
        for suf, ret in rets.items():
            ps = ret.retrieve(full_q)
            rec[f"passages{suf}"] = [{"title": t, "text": x, "sim": None} for t, x in ps]
            rec[f"failed{suf}"] = bool(ret.last_failed)
        done[key] = rec
        n_new += 1
        if n_new % 50 == 0:
            out.write_text(json.dumps(done))
            st = {s or "title": (x.n_errors, x.n_throttled) for s, x in rets.items()}
            print(f"[{split}] {len(done)}/{len(df)} | (errors, throttled)={st} | "
                  f"{(time.time()-t0)/n_new:.2f}s/q", flush=True)
    out.write_text(json.dumps(done))
    failed = sum(1 for v in done.values() if any(v[f"failed{s}"] for s in MODES))
    print(f"[{split}] DONE {len(done)} rows | failed rows={failed} | "
          + " | ".join(f"with{s or '_title'}={sum(1 for v in done.values() if v['passages'+s])}" for s in MODES),
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", required=True, choices=list(SPLIT_FILES))
    ap.add_argument("--out_dir", type=Path, default=REPO / "outputs" / "retrieval_v2")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for sp in a.split:
        run(sp, a.out_dir)
