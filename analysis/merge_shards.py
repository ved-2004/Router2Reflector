"""Merge a main run's partial JSONL (rows [0, cut)) with row-range shard CSVs into one {split}.csv.

  python analysis/merge_shards.py --partial <train.partial.jsonl> --cut 1300 \
      --shards outputs/pipeline_runs/qwen2.5-32B/shards/train.rows*.csv \
      --out outputs/pipeline_runs/qwen2.5-32B/train.csv --n 2400
"""
import argparse
import json

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--partial", required=True)
ap.add_argument("--cut", type=int, required=True)
ap.add_argument("--shards", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, required=True)
a = ap.parse_args()

main = pd.DataFrame([json.loads(l) for l in open(a.partial)]).drop_duplicates("row_id", keep="last")
main = main[main.row_id < a.cut]
df = pd.concat([main] + [pd.read_csv(f) for f in a.shards], ignore_index=True)
df = df.drop_duplicates("row_id", keep="first").sort_values("row_id").reset_index(drop=True)
assert list(df.row_id) == list(range(a.n)), f"rows missing/extra: have {len(df)}, expected {a.n}"
df["source_run"] = ["main" if r < a.cut else "h100_shard" for r in df.row_id]
df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(df)} rows ({(df.source_run == 'main').sum()} main, {(df.source_run != 'main').sum()} shard)")
