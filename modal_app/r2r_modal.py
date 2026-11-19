"""
R2R combined-pipeline data collection on Modal.

Runs every component on EVERY row so that any Router x RAG x Reflector combination can be
simulated offline afterwards (same trick as ``always_reflect_once`` in the Reflector grid).

Stages (run from the Router2Reflector/ directory):

  modal run --detach modal_app/r2r_modal.py --stage prep  --splits validation,test
  modal run --detach modal_app/r2r_modal.py --stage infer --splits validation,test --models 3B,7B,14B,32B
  modal volume get r2r-results runs outputs/pipeline_runs      # pull results locally

Per (model, split) the output CSV has, for each condition below, columns
``{cond}_pred, {cond}_correct, {cond}_H, {cond}_margin, {cond}_top1, {cond}_probs``:

  direct                       label-only direct pass (identical to Reflector grid / R2R vanilla)
  refl_<style> (x4)            reflection pass, no retrieval (identical to always_reflect_once)
  rag_raw / rag_filt           direct prompt + retrieved Wikipedia context (unfiltered / SBERT>=0.3)
  er_raw / er_filt             evidence_reconstruction reflection prompt + retrieved context

Context conditions are only run when that context is non-empty (``has_raw`` / ``has_filt``);
otherwise their columns are NaN and the pipeline falls back to the direct answer.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "bitsandbytes==0.45.0",
        "sentence-transformers==3.3.1",
        "wikipedia-api==0.7.1",
        "requests==2.32.3",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "huggingface_hub==0.26.2",
        "hf_transfer==0.1.8",
        "sentencepiece==0.2.0",
        "safetensors==0.4.5",
        "pyarrow==18.0.0",
        "bm25s==0.2.5",
        "PyStemmer==2.2.0.3",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/src"})
    .add_local_dir(REPO / "src", remote_path="/root/src")
    .add_local_dir(REPO / "data" / "raw", remote_path="/root/data")
)

app = modal.App("r2r-pipeline", image=image)
hf_vol = modal.Volume.from_name("r2r-hf-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("r2r-results", create_if_missing=True)
wiki_vol = modal.Volume.from_name("r2r-wiki", create_if_missing=True)
WIKI_SNAPSHOT = "20231101.en"   # HF wikimedia/wikipedia config used for the offline index

MODELS = {
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
    "32B": "Qwen/Qwen2.5-32B-Instruct",
}
# No H100s (cost). A10G for 3B/7B, L40S for 14B/32B -- the configuration of the first full pass.
GPU_FOR = {"3B": "A10G", "7B": "A10G", "14B": "L40S", "32B": "L40S"}
SPLIT_FILES = {"train": "Train.csv", "validation": "Validation.csv", "test": "Test.csv"}
SIMILARITY_THRESHOLD = 0.3   # same as r2r_baseline_comparison.py
TOP_K_PASSAGES = 3
MAX_CONTEXT_CHARS = 1500


def load_split(split: str):
    import ast
    import pandas as pd
    from reflector_core import get_passage_for_row, normalize_label

    df = pd.read_csv(f"/root/data/{SPLIT_FILES[split]}").reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        labels = ast.literal_eval(r["choice_labels"])
        texts = ast.literal_eval(r["choice_texts"])
        rows.append({
            "row_id": int(i),
            "source": str(r["source"]),
            "question": str(r["question"]),
            "options": dict(zip(labels, texts)),
            "passage": get_passage_for_row(str(r["source"]), r.get("context")),
            "gold": normalize_label(str(r["gold_label"])),
        })
    return rows


# ----------------------------------------------------------------------------------------------
# Stage 1 (CPU): retrieval (model-independent, done once per split) and weight download
# ----------------------------------------------------------------------------------------------
RETRIEVAL_MODES = {"": "opensearch", "_ft": "fulltext"}   # suffix -> RAGRetriever search_mode


def _retrieve_rows(split: str, retrievers: dict, limit: int = 0) -> dict:
    """Shared loop: retrieve with every retriever, add SBERT sims, write retrieval_v2/{split}.json.
    Rows whose requests failed are flagged and re-tried on the next call."""
    import time
    from sentence_transformers import SentenceTransformer

    out_path = Path(f"/results/retrieval_v2/{split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else {}

    rows = load_split(split)
    rows = rows[:limit] if limit else rows
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    t0 = time.time()
    for k, row in enumerate(rows):
        key = str(row["row_id"])
        if key in done and not any(done[key].get(f"failed{suf}") for suf in RETRIEVAL_MODES):
            continue
        # Same query construction as r2r_baseline_comparison.py
        full_q = row["question"] if not row["passage"] else f"{row['passage']}\n\n{row['question']}"
        q_emb = None
        rec = {"queries": retrievers[""].extract_queries(full_q)}
        for suf, retriever in retrievers.items():
            passages = retriever.retrieve(full_q)
            sims = []
            if passages:
                if q_emb is None:
                    q_emb = embed.encode(full_q[:512], normalize_embeddings=True)
                p_emb = embed.encode([text[:300] for _, text in passages], normalize_embeddings=True)
                sims = [float((q_emb * e).sum()) for e in p_emb]
            rec[f"passages{suf}"] = [{"title": t, "text": x, "sim": s} for (t, x), s in zip(passages, sims)]
            rec[f"failed{suf}"] = bool(retriever.last_failed)
        done[key] = rec
        if (k + 1) % 100 == 0:
            out_path.write_text(json.dumps(done))
            res_vol.commit()
            errs = {suf or "title": r.n_errors for suf, r in retrievers.items()}
            print(f"[{split}] {k+1}/{len(rows)} retrieved | errors={errs} | {time.time()-t0:.0f}s")
    out_path.write_text(json.dumps(done))
    res_vol.commit()
    summary = {"split": split, "rows": len(done)}
    for suf in RETRIEVAL_MODES:
        summary[f"with_raw{suf}"] = sum(1 for v in done.values() if v[f"passages{suf}"])
        summary[f"with_filtered{suf}"] = sum(
            1 for v in done.values() if any(p["sim"] >= SIMILARITY_THRESHOLD for p in v[f"passages{suf}"]))
        summary[f"failed_rows{suf}"] = sum(1 for v in done.values() if v[f"failed{suf}"])
    return summary


@app.function(volumes={"/results": res_vol}, cpu=2.0, memory=4096, timeout=12 * 3600)
def retrieve(split: str, limit: int = 0) -> dict:
    """LIVE Wikipedia API retrieval (rate-limited to ~10 req/min per IP as of Nov 2026; see README)."""
    from rag_pipeline import RAGRetriever
    retrievers = {suf: RAGRetriever(top_k=TOP_K_PASSAGES, search_mode=m) for suf, m in RETRIEVAL_MODES.items()}
    return _retrieve_rows(split, retrievers, limit)


@app.function(volumes={"/results": res_vol, "/wiki": wiki_vol}, cpu=4.0, memory=65536, timeout=6 * 3600)
def retrieve_offline(split: str, limit: int = 0) -> dict:
    """OFFLINE retrieval from the Wikipedia snapshot index built by build_wiki_index."""
    import copy
    from rag_pipeline import OfflineWikiRetriever
    ft = OfflineWikiRetriever("/wiki/index", top_k=TOP_K_PASSAGES, search_mode="fulltext")
    title = copy.copy(ft)                    # share the loaded corpus; title-prefix mode
    title.search_mode = "opensearch"
    return _retrieve_rows(split, {"": title, "_ft": ft}, limit)


@app.function(volumes={"/wiki": wiki_vol, "/hf": hf_vol}, cpu=16.0, memory=131072, timeout=8 * 3600)
def build_wiki_index() -> dict:
    """Build the offline index from HF wikimedia/wikipedia (WIKI_SNAPSHOT):
      docs.pkl : titles, leads (= text[:600], the lead-section passage), lengths, sorted lowercase titles
      bm25/    : bm25s index over "title. lead" (English stopwords + Snowball stemming)"""
    import glob
    import pickle
    import time
    import bm25s
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import Stemmer
    from huggingface_hub import snapshot_download

    t0 = time.time()
    local = snapshot_download("wikimedia/wikipedia", repo_type="dataset",
                              allow_patterns=[f"{WIKI_SNAPSHOT}/*.parquet"])
    files = sorted(glob.glob(f"{local}/{WIKI_SNAPSHOT}/*.parquet"))
    hf_vol.commit()
    print(f"downloaded {len(files)} parquet files in {time.time()-t0:.0f}s")
    titles, leads, lengths = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["title", "text"])
        titles += t.column("title").to_pylist()
        leads += pc.utf8_slice_codeunits(t.column("text"), 0, 600).to_pylist()
        lengths.append(pc.utf8_length(t.column("text")).to_numpy(zero_copy_only=False))
        del t
    lengths = np.concatenate(lengths).astype(np.int32)
    print(f"loaded {len(titles)} articles in {time.time()-t0:.0f}s")

    lower = [x.lower() for x in titles]
    order = np.array(sorted(range(len(lower)), key=lower.__getitem__), dtype=np.int32)
    sorted_lower = [lower[i] for i in order]
    del lower
    out = Path("/wiki/index")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "docs.pkl", "wb") as fh:
        pickle.dump({"titles": titles, "leads": leads, "lengths": lengths,
                     "sorted_lower": sorted_lower, "order": order, "snapshot": WIKI_SNAPSHOT}, fh, protocol=5)
    wiki_vol.commit()
    print(f"saved docs.pkl in {time.time()-t0:.0f}s")

    stemmer = Stemmer.Stemmer("english")
    corpus = [f"{a}. {b}" for a, b in zip(titles, leads)]
    del sorted_lower
    tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, show_progress=False)
    del corpus
    print(f"tokenized in {time.time()-t0:.0f}s | vocab={len(tokens.vocab)}")
    bm = bm25s.BM25()
    bm.index(tokens, show_progress=False)
    bm.save(str(out / "bm25"))
    wiki_vol.commit()
    return {"articles": len(titles), "vocab": len(tokens.vocab), "seconds": round(time.time() - t0)}


@app.function(volumes={"/results": res_vol}, cpu=4.0, memory=8192, timeout=2 * 3600)
def score(split: str) -> dict:
    """Fill SBERT similarities into a retrieval_v2 file fetched locally (local_retrieve.py, sim=null).
    Same encoding as ``retrieve`` / rag_pipeline.SimilarityFilter: question[:512] vs passage[:300]."""
    from sentence_transformers import SentenceTransformer

    path = Path(f"/results/retrieval_v2/{split}.json")
    data = json.loads(path.read_text())
    rows = {str(r["row_id"]): r for r in load_split(split)}
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    n = 0
    for key, rec in data.items():
        row = rows[key]
        full_q = row["question"] if not row["passage"] else f"{row['passage']}\n\n{row['question']}"
        todo = [p for suf in RETRIEVAL_MODES for p in rec[f"passages{suf}"] if p["sim"] is None]
        if not todo:
            continue
        q_emb = embed.encode(full_q[:512], normalize_embeddings=True)
        p_emb = embed.encode([p["text"][:300] for p in todo], normalize_embeddings=True)
        for p, e in zip(todo, p_emb):
            p["sim"] = float((q_emb * e).sum())
        n += len(todo)
    path.write_text(json.dumps(data))
    res_vol.commit()
    summary = {"split": split, "rows": len(data), "scored_passages": n}
    for suf in RETRIEVAL_MODES:
        summary[f"with_raw{suf}"] = sum(1 for v in data.values() if v[f"passages{suf}"])
        summary[f"with_filtered{suf}"] = sum(
            1 for v in data.values() if any(p["sim"] >= SIMILARITY_THRESHOLD for p in v[f"passages{suf}"]))
        summary[f"failed_rows{suf}"] = sum(1 for v in data.values() if v[f"failed{suf}"])
    return summary


@app.function(volumes={"/hf": hf_vol}, cpu=4.0, memory=8192, timeout=3 * 3600)
def download(model_id: str) -> str:
    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    hf_vol.commit()
    return path


# ----------------------------------------------------------------------------------------------
# Stage 2 (GPU): every condition on every row
# ----------------------------------------------------------------------------------------------
def build_context(passages, min_sim=None):
    keep = [p for p in passages if min_sim is None or p["sim"] >= min_sim]
    if not keep:
        return ""
    return "\n\n".join(f"[{p['title']}]\n{p['text']}" for p in keep)[:MAX_CONTEXT_CHARS]


@app.function(volumes={"/hf": hf_vol, "/results": res_vol}, gpu="A10G", memory=32768, timeout=12 * 3600)
def infer(size: str, split: str, conditions: list[str] | None = None, limit: int = 0,
          row_start: int = 0, row_end: int = 0, shard: str = "") -> dict:
    """row_start/row_end + shard: process only rows [row_start, row_end) into separate
    ``{split}.{shard}.*`` files, so extra containers can work on part of a split in parallel."""
    import time
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import reflector_core as rc

    model_id = MODELS[size]
    conditions = conditions or (["direct"] + [f"refl_{s}" for s in rc.PROMPT_STYLE_ORDER]
                                + [f"{kind}{suf}_{flt}" for suf in RETRIEVAL_MODES
                                   for kind in ("rag", "er") for flt in ("raw", "filt")])
    retrieval = json.loads(Path(f"/results/retrieval_v2/{split}.json").read_text())
    rows = load_split(split)
    rows = rows[:limit] if limit else rows
    if row_end:
        rows = rows[row_start:row_end]
    stem = f"{split}.{shard}" if shard else split

    out_dir = Path(f"/results/runs_v2/qwen2.5-{size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"{stem}.partial.jsonl"
    done = {}
    if part_path.exists():
        for line in part_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["row_id"]] = rec

    # Same 4-bit config as the Reflector notebook and r2r_baseline_comparison.py
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant,
                                               trust_remote_code=True)
    mdl.eval()
    print(f"Loaded {model_id} on {torch.cuda.get_device_name(0)} | {len(rows)} rows | {len(done)} cached")

    def run(messages, options):
        o = rc.infer_label_only(messages, options, mdl, tok)
        return {"pred": rc.normalize_label(o["predicted_by_probs"]), "H": o["entropy_over_labels"],
                "margin": o["margin"], "top1": o["top1_prob"], "probs": rc.safe_json_dumps(o["label_probs"])}

    t0 = time.time()
    fh = open(part_path, "a")
    for k, row in enumerate(rows):
        if row["row_id"] in done:
            continue
        q, opts, psg, gold = row["question"], row["options"], row["passage"], row["gold"]
        rr = retrieval[str(row["row_id"])]
        rec = {"row_id": row["row_id"], "source": row["source"], "gold": gold}
        ctxs = {}
        for suf in RETRIEVAL_MODES:
            passages = rr[f"passages{suf}"]
            ctxs[f"{suf}_raw"] = build_context(passages)
            ctxs[f"{suf}_filt"] = build_context(passages, SIMILARITY_THRESHOLD)
            rec.update({f"n_raw{suf}": len(passages),
                        f"n_filt{suf}": sum(p["sim"] >= SIMILARITY_THRESHOLD for p in passages),
                        f"s_max{suf}": max([p["sim"] for p in passages], default=np.nan),
                        f"has_raw{suf}": bool(ctxs[f"{suf}_raw"]), f"has_filt{suf}": bool(ctxs[f"{suf}_filt"]),
                        f"retrieval_failed{suf}": bool(rr[f"failed{suf}"])})
        for cond in conditions:
            if cond == "direct":
                r = run(rc.build_messages_direct(q, opts, psg), opts)
            elif cond.startswith("refl_"):
                r = run(rc.build_messages_reflection(q, opts, cond[5:], psg), opts)
            elif cond.startswith(("rag", "er")):
                kind, rest = cond.split("_", 1)            # "rag_ft_filt" -> ("rag", "ft_filt")
                ctx = ctxs["_" + rest]                     # keys: _raw, _filt, _ft_raw, _ft_filt
                if kind == "rag":
                    r = run(rc.build_messages_direct(q, opts, psg, ctx), opts) if ctx else None
                else:
                    r = run(rc.build_messages_reflection_with_context(
                        q, opts, "evidence_reconstruction", psg, ctx), opts) if ctx else None
            else:
                raise ValueError(cond)
            if r is not None:
                for kk, vv in r.items():
                    rec[f"{cond}_{kk}"] = vv
                rec[f"{cond}_correct"] = int(r["pred"] == gold)
        fh.write(json.dumps(rec) + "\n")
        done[row["row_id"]] = rec
        if (k + 1) % 50 == 0:
            fh.flush()
            res_vol.commit()
            print(f"[{size}/{split}] {k+1}/{len(rows)} | {time.time()-t0:.0f}s")
    fh.close()

    df = pd.DataFrame([done[r["row_id"]] for r in rows])
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    res_vol.commit()
    summary = {"size": size, "split": split, "n": len(df), "seconds": round(time.time() - t0)}
    for cond in conditions:
        col = f"{cond}_correct"
        if col in df:
            summary[f"{cond}_acc_on_run"] = round(float(df[col].mean()), 4)
            summary[f"{cond}_n"] = int(df[col].notna().sum())
    return summary


@app.local_entrypoint()
def main(stage: str = "prep", splits: str = "validation,test", models: str = "3B,7B,14B,32B",
         conditions: str = "", limit: int = 0, row_start: int = 0, row_end: int = 0, n_shards: int = 1,
         gpu: str = ""):
    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    size_list = [m.strip() for m in models.split(",") if m.strip()]
    if stage == "prep":
        handles = [retrieve.spawn(s, limit) for s in split_list] + [download.spawn(MODELS[m]) for m in size_list]
        for h in handles:
            print(h.get())
    elif stage == "build_index":
        print(build_wiki_index.remote())
    elif stage == "retrieve_offline":
        for h in [retrieve_offline.spawn(s, limit) for s in split_list]:
            print(h.get())
    elif stage == "score":
        for h in [score.spawn(s) for s in split_list]:
            print(h.get())
    elif stage == "infer":
        conds = [c for c in conditions.split(",") if c] or None
        calls = []
        for m in size_list:
            fn = infer.with_options(gpu=gpu or GPU_FOR[m])
            for s in split_list:
                if row_end:   # split [row_start, row_end) into n_shards parallel containers
                    edges = [row_start + round(i * (row_end - row_start) / n_shards) for i in range(n_shards + 1)]
                    for a, b in zip(edges[:-1], edges[1:]):
                        calls.append(fn.spawn(m, s, conds, 0, a, b, f"rows{a}-{b}"))
                else:
                    calls.append(fn.spawn(m, s, conds, limit))
        for c in calls:
            print(json.dumps(c.get()))
    else:
        raise ValueError(stage)
