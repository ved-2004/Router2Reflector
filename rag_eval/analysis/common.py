"""Shared local helpers: load splits, build retrieval configs from the candidate files pulled from the
r2r-rag-eval volume (rag_eval/results/retrieval/{corpus}/cands_*.json, passages_*.json)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # Router2Reflector/
EVAL = ROOT / "rag_eval"
RES = EVAL / "results"
SPLIT_FILES = {"train": "Train.csv", "validation": "Validation.csv", "test": "Test.csv"}
SIZES = ["3B", "7B", "14B", "32B"]
MAX_CONTEXT_CHARS = 1500
FAMILY = {"wiki": "R1", "med": "R3", "medtb": "R3", "arc": "R4"}


def load_split(split):
    df = pd.read_csv(ROOT / "data" / "raw" / SPLIT_FILES[split]).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        labels = ast.literal_eval(r["choice_labels"])
        texts = ast.literal_eval(r["choice_texts"])
        rows.append({"row_id": i, "source": r["source"], "question": str(r["question"]),
                     "options": dict(zip(labels, texts)), "gold": str(r["gold_label"]).strip().upper(),
                     "context": None if pd.isna(r["context"]) else str(r["context"])})
    return rows


def load_direct(size, split):
    return pd.read_csv(ROOT / "outputs" / "pipeline_runs" / f"qwen2.5-{size}" / f"{split}.csv")


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(s).lower()).split())


def build_context(passages, k, max_chars=MAX_CONTEXT_CHARS):
    keep = passages[:k]
    if not keep:
        return ""
    blocks = [f"[{p['title']}]\n{p['text']}" if p.get("title") else p["text"] for p in keep]
    return "\n\n".join(blocks)[:max_chars]


def _dedupe(lst):
    seen, out = set(), []
    for x in lst:
        pid = (x[1], x[2])
        if pid not in seen:
            seen.add(pid)
            out.append(x)
    return out


def configs_for_corpus(corpus, tag="validation_test", top=10):
    """Return {config_name: {split: {row_id: [passage dicts]}}} built from saved candidate lists."""
    d = RES / "retrieval" / corpus
    cands = json.loads((d / f"cands_{tag}.json").read_text())
    P = json.loads((d / f"passages_{tag}.json").read_text())
    by_row = {}
    for key, v in cands.items():
        sp, rid, lab = key.split("|")
        by_row.setdefault((sp, int(rid)), {})[lab] = v

    def mat(lst, n=top):
        out = []
        for s, p, l in lst[:n]:
            d0 = P[f"{p}:{l}"]
            out.append({"title": d0["title"], "text": d0["text"], "src": d0["src"], "score": s})
        return out

    pre = corpus
    cfg = {f"{pre}_{n}": {} for n in ["bm25", "dense", "bm25_rr", "dense_rr", "hybrid_rr",
                                      "bm25_opt", "dense_opt", "dense_opt_rr", "hybrid_opt_rr"]}
    for (sp, rid), qd in by_row.items():
        base = qd[""]
        ce = {(p, l): s for s, p, l in base["ce"]}
        put = lambda name, lst: cfg[name].setdefault(sp, {}).__setitem__(rid, mat(lst))
        put(f"{pre}_bm25", base["bm25"])
        put(f"{pre}_dense", base["dense"])
        put(f"{pre}_bm25_rr", sorted([[ce[(p, l)], p, l] for _, p, l in base["bm25"]], key=lambda x: -x[0]))
        put(f"{pre}_dense_rr", sorted([[ce[(p, l)], p, l] for _, p, l in base["dense"]], key=lambda x: -x[0]))
        put(f"{pre}_hybrid_rr", base["ce"])
        opts = [qd[k] for k in sorted(qd) if k != ""]
        for kind in ("bm25", "dense"):
            rr = []
            for rank in range(max(len(o[kind]) for o in opts)):
                for o in opts:
                    if rank < len(o[kind]):
                        rr.append(o[kind][rank])
            put(f"{pre}_{kind}_opt", _dedupe(rr))
        # option-aware + CE (query = question + option); global sort by CE score, dedupe
        allce = sorted([x for o in opts for x in o["ce"]], key=lambda x: -x[0])
        put(f"{pre}_hybrid_opt_rr", _dedupe(allce))
        dn = []
        for o in opts:
            ceo = {(p, l): s for s, p, l in o["ce"]}
            dn += [[ceo[(p, l)], p, l] for _, p, l in o["dense"]]
        put(f"{pre}_dense_opt_rr", _dedupe(sorted(dn, key=lambda x: -x[0])))
    return cfg


def r0_configs():
    out = {"r0_ft_raw": {}, "r0_ft_filt": {}}
    for sp in ("validation", "test"):
        data = json.loads((RES / "r0" / f"{sp}.json").read_text())
        for rid, rec in data.items():
            ps = rec.get("passages_ft") or []
            out["r0_ft_raw"].setdefault(sp, {})[int(rid)] = ps
            out["r0_ft_filt"].setdefault(sp, {})[int(rid)] = [p for p in ps if p["sim"] is not None and p["sim"] >= 0.3]
    return out
