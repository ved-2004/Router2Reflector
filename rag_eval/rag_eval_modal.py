"""
R2R replacement-RAG evaluation on Modal (app ``r2r-rag-eval``, volume ``r2r-rag-eval``).

Evaluation only: nothing under src/, modal_app/, analysis/, outputs/, data/ is modified, and the
existing volumes (r2r-hf-cache, r2r-wiki, r2r-results) are mounted READ-ONLY.

Corpora (built into volume r2r-rag-eval under /ev/corpora/{corpus}/part-XXXXX.parquet):
  wiki : full-article ~100-word chunks of HF wikimedia/wikipedia 20231101.en (cached parquet)  [R1/R2]
  med  : MedRAG textbooks (part 0) + StatPearls (part 1, built from NCBI with MedRAG's chunker)
         + MedRAG PubMed snippets (parts 2..51)                                                [R3]
  arc  : AI2 ARC_Corpus.txt (14.6M science sentences)                                          [R4]
Indexes: exact BM25 (Lucene idf, k1=1.5, b=0.75, global statistics across parts; hashed vocab 2^24,
         bm25s English stopwords + Snowball stemming) and dense (BAAI/bge-base-en-v1.5 for wiki/arc,
         ncbi/MedCPT article/query encoders for med), fp16 embeddings, exact GPU matmul search.
Rerankers: BAAI/bge-reranker-base (wiki/arc), ncbi/MedCPT-Cross-Encoder (med).

Run from Router2Reflector/ (each stage is a remote orchestrator, so --detach survives disconnects):
  modal run --detach rag_eval/rag_eval_modal.py --stage prep          # build corpora (CPU fan-out)
  modal run --detach rag_eval/rag_eval_modal.py --stage index         # BM25 parts + dense embeddings
  modal run --detach rag_eval/rag_eval_modal.py --stage retrieve --splits validation,test
  modal run --detach rag_eval/rag_eval_modal.py --stage generate      # R5 background generation
  modal run --detach rag_eval/rag_eval_modal.py --stage contexts      # build LLM context files
  modal run --detach rag_eval/rag_eval_modal.py --stage infer --options a,b,c
  modal run --detach rag_eval/rag_eval_modal.py --stage nli           # answer-support gate scores
  modal run --detach rag_eval/rag_eval_modal.py --stage contam        # contamination scan
  modal run --detach rag_eval/rag_eval_modal.py --stage r3            # R3 MedRAG end to end (resumable)
  modal volume get r2r-rag-eval /retrieval rag_eval/results/  (etc.; see rag_eval/pull.sh)
Then locally: rag_eval/analysis/stage1_recall.py and rag_eval/analysis/stage2_analysis.py.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SMALL_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-reranker-base",
    "ncbi/MedCPT-Query-Encoder",
    "ncbi/MedCPT-Article-Encoder",
    "ncbi/MedCPT-Cross-Encoder",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]


def _download_small_models():
    from huggingface_hub import snapshot_download
    for m in SMALL_MODELS:
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.bin"],
                          ignore_patterns=["*onnx*", "*.msgpack", "*.h5", "*tf_model*", "*flax*"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "bitsandbytes==0.45.0",
        "sentence-transformers==3.3.1",
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
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "protobuf==5.28.3",
        "tiktoken==0.8.0",
    )
    # small encoders/rerankers baked into the image (HF_HOME=/root/hfsmall); Qwen weights come from
    # the read-only r2r-hf-cache volume via cache_dir=/hf/hub.
    .env({"HF_HOME": "/root/hfsmall", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_small_models)
    .env({"PYTHONPATH": "/root/src:/root/rag_eval", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(REPO / "src", remote_path="/root/src")
    .add_local_dir(REPO / "data" / "raw", remote_path="/root/data")
    .add_local_file(HERE / "statpearls_chunk.py", remote_path="/root/rag_eval/statpearls_chunk.py")
    .add_local_dir(HERE / "analysis", remote_path="/root/rag_eval/analysis", ignore=["__pycache__"])
)

app = modal.App("r2r-rag-eval", image=image)
ev_vol = modal.Volume.from_name("r2r-rag-eval", create_if_missing=True, version=2)
hf_ro = modal.Volume.from_name("r2r-hf-cache").read_only()          # Qwen weights + wiki parquet
res_ro = modal.Volume.from_name("r2r-results").read_only()          # R0 retrieval_v2 (read only)

EV = Path("/ev")
WIKI_PARQUET = "/hf/hub/datasets--wikimedia--wikipedia/snapshots/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en"
MODELS = {
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
    "32B": "Qwen/Qwen2.5-32B-Instruct",
}
SPLIT_FILES = {"train": "Train.csv", "validation": "Validation.csv", "test": "Test.csv"}
MAX_CONTEXT_CHARS = 1500
N_FEATURES = 2 ** 24
BM25_K1, BM25_B = 1.5, 0.75
DENSE_ENCODER = {"wiki": "bge", "arc": "bge", "med": "medcpt"}
CE_MODEL = {"wiki": "BAAI/bge-reranker-base", "arc": "BAAI/bge-reranker-base", "med": "ncbi/MedCPT-Cross-Encoder"}
BGE_QUERY_INSTR = "Represent this sentence for searching relevant passages: "
N_PUBMED_GROUPS = 59          # 1166 MedRAG pubmed chunk files (pubmed23n0001..1166) -> 59 parts of <=20 files
# The workspace has a hard cap of 10 concurrent GPUs. Each GPU function gets an explicit container cap
# and stages are run one at a time (phase order), so the total stays <= 10. CPU work fans out freely.
GPU_BUDGET = {"embed": 4, "rerank": 3, "llm": 4, "nli": 2}
# Budget GPUs (no H100s): L4 (24 GB) for encoders, rerankers, NLI and Qwen 3B-14B; L40S (48 GB) for Qwen-32B,
# whose NF4 weights (~20 GB incl. fp16 embed/lm_head) are too tight for an L4 without CPU offload.
GPU_SMALL, GPU_BIG = "L4", "L40S"
BIG_SIZES = {"32B"}
VOLS_RW = {"/ev": ev_vol}
VOLS_ALL = {"/ev": ev_vol, "/hf": hf_ro}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def load_split(split: str):
    """Identical to modal_app/r2r_modal.py::load_split."""
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
            "raw_context": None if str(r.get("context")) in ("nan", "None", "") else str(r.get("context")),
        })
    return rows


_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_ANALYZER = {}


def analyze(text: str):
    """BM25 analyzer: lowercase, \\w\\w+ tokens, bm25s English stopwords, Snowball stemming."""
    if not _ANALYZER:
        import Stemmer
        from bm25s.stopwords import STOPWORDS_EN
        _ANALYZER["stop"] = set(STOPWORDS_EN)
        _ANALYZER["stem"] = Stemmer.Stemmer("english")
    stop = _ANALYZER["stop"]
    return _ANALYZER["stem"].stemWords([t for t in _TOKEN_RE.findall(text.lower()) if t not in stop])


def hasher():
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(n_features=N_FEATURES, analyzer=analyze, alternate_sign=False, norm=None,
                             dtype=__import__("numpy").float32)


def part_paths(corpus: str):
    return sorted((EV / "corpora" / corpus).glob("part-*.parquet"))


def doc_string(title: str, text: str) -> str:
    title = (title or "").strip()
    return f"{title}. {text}" if title else text


def build_context(passages, k: int, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Same format as modal_app/r2r_modal.py::build_context ([title]\\ntext, blank-line joined, 1500 chars)."""
    keep = passages[:k]
    if not keep:
        return ""
    blocks = [f"[{p['title']}]\n{p['text']}" if p.get("title") else p["text"] for p in keep]
    return "\n\n".join(blocks)[:max_chars]


def _download(url: str, dest: str):
    import requests
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 22):
                fh.write(chunk)


def _write_parquet(path: Path, cols: dict):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), str(path), compression="zstd")


# ----------------------------------------------------------------------------------------------
# Corpus preparation (CPU fan-out)
# ----------------------------------------------------------------------------------------------
@app.function(volumes=VOLS_ALL, cpu=4.0, memory=32768, timeout=3 * 3600, retries=2)
def prep_wiki(i: int) -> dict:
    """Chunk Wikipedia parquet file i into ~100-word passages (DPR style; a trailing chunk shorter
    than 20 words is appended to the previous chunk)."""
    import pyarrow.parquet as pq
    t0 = time.time()
    f = sorted(Path(WIKI_PARQUET).glob("*.parquet"))[i]
    t = pq.read_table(str(f), columns=["title", "text"])
    titles_in, texts_in = t.column("title").to_pylist(), t.column("text").to_pylist()
    del t
    titles, texts = [], []
    for ti, tx in zip(titles_in, texts_in):
        w = tx.split()
        chunks = [w[j:j + 100] for j in range(0, len(w), 100)]
        if len(chunks) > 1 and len(chunks[-1]) < 20:
            chunks[-2] = chunks[-2] + chunks[-1]
            chunks.pop()
        for c in chunks:
            titles.append(ti)
            texts.append(" ".join(c))
    _write_parquet(EV / "corpora" / "wiki" / f"part-{i:05d}.parquet",
                   {"title": titles, "text": texts, "src": ["wiki"] * len(texts)})
    ev_vol.commit()
    return {"part": i, "articles": len(titles_in), "passages": len(texts), "s": round(time.time() - t0)}


@app.function(volumes=VOLS_RW, cpu=4.0, memory=16384, timeout=3 * 3600, retries=2)
def prep_med(part: int) -> dict:
    """part 0 = MedRAG textbooks, 1 = StatPearls (NCBI + MedRAG chunker), 2.. = PubMed groups."""
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    titles, texts, srcs = [], [], []
    if part == 0:
        from huggingface_hub import list_repo_files
        files = [f for f in list_repo_files("MedRAG/textbooks", repo_type="dataset") if f.endswith(".jsonl")]
        for fn in sorted(files):
            p = hf_hub_download("MedRAG/textbooks", fn, repo_type="dataset", cache_dir="/tmp/hf")
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    titles.append(r["title"]); texts.append(r["content"]); srcs.append("textbooks")
    elif part == 1:
        import tarfile
        _download("https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz", "/tmp/sp/sp.tgz")
        with tarfile.open("/tmp/sp/sp.tgz") as tf:
            tf.extractall("/tmp/sp/x", filter="data")
        from statpearls_chunk import extract
        fns = sorted(Path("/tmp/sp/x").rglob("*.nxml"))
        for fp in fns:
            try:
                for s in extract(str(fp)):
                    r = json.loads(s)
                    titles.append(r["title"]); texts.append(r["content"]); srcs.append("statpearls")
            except Exception as e:  # malformed file: skip, as MedRAG effectively does
                print("skip", fp.name, e)
    else:
        g = part - 2
        for n in range(g * 20 + 1, min(g * 20 + 21, 1167)):
            fn = f"chunk/pubmed23n{n:04d}.jsonl"
            p = hf_hub_download("MedRAG/pubmed", fn, repo_type="dataset", cache_dir="/tmp/hf")
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    titles.append(r["title"]); texts.append(r["content"]); srcs.append("pubmed")
            Path(p).unlink(missing_ok=True)
    _write_parquet(EV / "corpora" / "med" / f"part-{part:05d}.parquet", {"title": titles, "text": texts, "src": srcs})
    ev_vol.commit()
    return {"part": part, "passages": len(texts), "s": round(time.time() - t0)}


@app.function(volumes=VOLS_RW, cpu=4.0, memory=32768, timeout=3 * 3600, retries=2)
def prep_arc() -> dict:
    """AI2 ARC Corpus (ARC-V1-Feb2018.zip, ai2-public-datasets S3) -> parts of 1M sentences."""
    import zipfile
    t0 = time.time()
    _download("https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip", "/tmp/arc/arc.zip")
    with zipfile.ZipFile("/tmp/arc/arc.zip") as z:
        name = [n for n in z.namelist() if n.endswith("ARC_Corpus.txt")][0]
        lines = z.read(name).decode("utf-8", errors="replace").splitlines()
    lines = [x.strip() for x in lines if x.strip()]
    n_parts = 0
    for k in range(0, len(lines), 1_000_000):
        chunk = lines[k:k + 1_000_000]
        _write_parquet(EV / "corpora" / "arc" / f"part-{n_parts:05d}.parquet",
                       {"title": [""] * len(chunk), "text": chunk, "src": ["arc"] * len(chunk)})
        n_parts += 1
    ev_vol.commit()
    return {"sentences": len(lines), "parts": n_parts, "s": round(time.time() - t0)}


# ----------------------------------------------------------------------------------------------
# BM25 (exact, global stats) -- per-part postings, CPU fan-out
# ----------------------------------------------------------------------------------------------
@app.function(volumes=VOLS_RW, cpu=8.0, memory=65536, timeout=4 * 3600, retries=2)
def bm25_part(corpus: str, part: int) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    t0 = time.time()
    t = pq.read_table(str(EV / "corpora" / corpus / f"part-{part:05d}.parquet"), columns=["title", "text"])
    docs = [doc_string(a, b) for a, b in zip(t.column("title").to_pylist(), t.column("text").to_pylist())]
    del t
    X = hasher().transform(docs)          # docs x 2^24 term counts (CSR)
    del docs
    dl = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    C = X.tocsc()
    del X
    out = EV / "bm25" / corpus / f"part-{part:05d}"
    out.mkdir(parents=True, exist_ok=True)
    df = np.diff(C.indptr).astype(np.int32)
    nz = np.nonzero(df)[0].astype(np.int32)
    np.save(out / "indptr.npy", C.indptr.astype(np.int64))
    np.save(out / "indices.npy", C.indices.astype(np.int32))
    np.save(out / "tf.npy", np.minimum(C.data, 65535).astype(np.uint16))
    np.save(out / "dl.npy", dl)
    np.save(out / "df_idx.npy", nz)
    np.save(out / "df_cnt.npy", df[nz])
    ev_vol.commit()
    return {"corpus": corpus, "part": part, "docs": int(len(dl)), "nnz": int(C.nnz), "s": round(time.time() - t0)}


@app.function(volumes=VOLS_RW, cpu=4.0, memory=32768, timeout=3600)
def bm25_global(corpus: str) -> dict:
    import numpy as np
    ev_vol.reload()
    df = np.zeros(N_FEATURES, dtype=np.int64)
    n_docs, tot_len = 0, 0.0
    for p in sorted((EV / "bm25" / corpus).glob("part-*")):
        idx, cnt = np.load(p / "df_idx.npy"), np.load(p / "df_cnt.npy")
        df[idx] += cnt
        dl = np.load(p / "dl.npy")
        n_docs += len(dl); tot_len += float(dl.sum())
    np.save(EV / "bm25" / corpus / "global_df.npy", df.astype(np.int32))
    meta = {"N": n_docs, "avgdl": tot_len / n_docs}
    (EV / "bm25" / corpus / "global.json").write_text(json.dumps(meta))
    ev_vol.commit()
    return {"corpus": corpus, **meta}


@app.function(volumes=VOLS_RW, cpu=4.0, memory=32768, timeout=2 * 3600, retries=2)
def bm25_search_part(corpus: str, part: int, q_terms: list, q_idf: list, avgdl: float, K: int) -> tuple:
    """Exact BM25 over one part with GLOBAL idf/avgdl passed in. Returns (scores[nq,K], ids[nq,K])."""
    import numpy as np
    d = EV / "bm25" / corpus / f"part-{part:05d}"
    indptr, indices = np.load(d / "indptr.npy"), np.load(d / "indices.npy")
    tf, dl = np.load(d / "tf.npy").astype(np.float32), np.load(d / "dl.npy")
    n = len(dl)
    norm = (BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)).astype(np.float32)
    S = np.full((len(q_terms), K), -1.0, dtype=np.float32)
    I = np.full((len(q_terms), K), -1, dtype=np.int64)
    for qi, (terms, idfs) in enumerate(zip(q_terms, q_idf)):
        segs_d, segs_w = [], []
        for h, w in zip(terms, idfs):
            a, b = indptr[h], indptr[h + 1]
            if b > a:
                docs = indices[a:b]
                t = tf[a:b]
                segs_d.append(docs)
                segs_w.append(w * t * (BM25_K1 + 1) / (t + norm[docs]))
        if not segs_d:
            continue
        sc = np.bincount(np.concatenate(segs_d), weights=np.concatenate(segs_w), minlength=n)
        kk = min(K, n)
        top = np.argpartition(-sc, kk - 1)[:kk]
        top = top[np.argsort(-sc[top])]
        top = top[sc[top] > 0]
        S[qi, :len(top)] = sc[top]
        I[qi, :len(top)] = top
    return S, I


# ----------------------------------------------------------------------------------------------
# Dense encoders (GPU)
# ----------------------------------------------------------------------------------------------
class _Enc:
    """bge (sentence-transformers, normalized, fp16) or MedCPT (CLS pooling, fp16, inner product)."""

    def __init__(self, kind: str, side: str):
        import torch
        self.kind, self.side = kind, side
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if self.dev == "cuda" else torch.float32
        if kind == "bge":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer("BAAI/bge-base-en-v1.5", device=self.dev)
            if self.dev == "cuda":
                self.m.half()
            self.m.max_seq_length = 256 if side == "doc" else 512
        else:
            from transformers import AutoModel, AutoTokenizer
            name = "ncbi/MedCPT-Article-Encoder" if side == "doc" else "ncbi/MedCPT-Query-Encoder"
            self.tok = AutoTokenizer.from_pretrained(name)
            self.m = AutoModel.from_pretrained(name, torch_dtype=dt).to(self.dev).eval()

    def encode(self, items, bs=512):
        """items: list of str (bge) or list of [title, text] / str (MedCPT)."""
        import numpy as np
        import torch
        if self.kind == "bge":
            if self.side == "query":
                items = [BGE_QUERY_INSTR + q for q in items]
            return self.m.encode(items, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float16)
        lens = np.array([len(x[1]) + len(x[0]) if isinstance(x, (list, tuple)) else len(x) for x in items])
        order = np.argsort(-lens)
        out = np.zeros((len(items), 768), dtype=np.float16)
        maxlen = 512
        for s in range(0, len(items), bs):
            idx = order[s:s + bs]
            batch = [items[i] for i in idx]
            enc = self.tok(batch, truncation=True, padding=True, max_length=maxlen, return_tensors="pt").to(self.dev)
            with torch.inference_mode():
                e = self.m(**enc).last_hidden_state[:, 0, :]
            out[idx] = e.float().cpu().numpy().astype(np.float16)
        return out


@app.function(volumes=VOLS_RW, gpu=GPU_SMALL, cpu=8.0, memory=65536, timeout=4 * 3600, retries=2,
              max_containers=GPU_BUDGET["embed"])
def embed_part(corpus: str, part: int) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    t0 = time.time()
    kind = DENSE_ENCODER[corpus]
    out = EV / "emb" / f"{corpus}_{kind}" / f"part-{part:05d}.npy"
    if out.exists():
        return {"corpus": corpus, "part": part, "skipped": True}
    t = pq.read_table(str(EV / "corpora" / corpus / f"part-{part:05d}.parquet"), columns=["title", "text"])
    titles, texts = t.column("title").to_pylist(), t.column("text").to_pylist()
    del t
    enc = _Enc(kind, "doc")
    items = ([doc_string(a, b) for a, b in zip(titles, texts)] if kind == "bge"
             else [[a, b] for a, b in zip(titles, texts)])
    del titles, texts
    embs = []
    for s in range(0, len(items), 200_000):
        embs.append(enc.encode(items[s:s + 200_000], bs=1024 if kind == "bge" else 512))
        print(f"{corpus}/{part}: {s + len(embs[-1])}/{len(items)} | {time.time() - t0:.0f}s", flush=True)
    E = np.concatenate(embs)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, E)
    ev_vol.commit()
    return {"corpus": corpus, "part": part, "n": int(len(E)), "s": round(time.time() - t0)}


@app.function(cpu=16.0, memory=32768, timeout=3600)
def encode_queries(kind: str, queries: list) -> "object":
    """Query encoding on CPU (fp32; ~3k queries) so it does not wait on the GPU cap."""
    import torch
    torch.set_num_threads(16)
    return _Enc(kind, "query").encode(queries, bs=64)


@app.function(volumes=VOLS_RW, cpu=16.0, memory=65536, timeout=2 * 3600, retries=2)
def dense_search_part(corpus: str, part: int, Q, K: int) -> tuple:
    """Exact inner-product search on CPU (fp32 matmul over the fp16-stored embeddings) -- keeps the
    10-GPU workspace cap free for encoders/LLMs. Embeddings are processed in 250k-row blocks."""
    import numpy as np
    import torch
    torch.set_num_threads(16)
    kind = DENSE_ENCODER[corpus]
    E16 = np.load(EV / "emb" / f"{corpus}_{kind}" / f"part-{part:05d}.npy", mmap_mode="r")
    Qt = torch.from_numpy(np.asarray(Q, dtype=np.float32))
    n = E16.shape[0]
    kk = min(K, n)
    bestS = torch.full((Qt.shape[0], kk), -1e9)
    bestI = torch.full((Qt.shape[0], kk), -1, dtype=torch.int64)
    for a in range(0, n, 250_000):
        Eb = torch.from_numpy(np.asarray(E16[a:a + 250_000], dtype=np.float32))
        sc = Qt @ Eb.T
        v, i = torch.topk(sc, min(kk, Eb.shape[0]), dim=1)
        catS, catI = torch.cat([bestS, v], 1), torch.cat([bestI, i + a], 1)
        top = torch.topk(catS, kk, dim=1)
        bestS, bestI = top.values, torch.gather(catI, 1, top.indices)
    return bestS.numpy(), bestI.numpy().astype(np.int64)


@app.function(volumes=VOLS_RW, cpu=4.0, memory=32768, timeout=3600, retries=2)
def fetch_texts(corpus: str, part: int, ids: list) -> dict:
    import pyarrow.parquet as pq
    t = pq.read_table(str(EV / "corpora" / corpus / f"part-{part:05d}.parquet"))
    sub = t.take(sorted(set(int(i) for i in ids))).to_pylist()
    keys = sorted(set(int(i) for i in ids))
    return {k: {"title": r["title"], "text": r["text"], "src": r["src"]} for k, r in zip(keys, sub)}


@app.function(gpu=GPU_SMALL, memory=32768, timeout=2 * 3600, retries=2, max_containers=GPU_BUDGET["rerank"])
def rerank(model_name: str, pairs: list) -> list:
    """Cross-encoder scores for (query, passage) pairs (bge-reranker-base or MedCPT-Cross-Encoder)."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
    lens = np.array([len(a) + len(b) for a, b in pairs])
    order = np.argsort(-lens)
    out = np.zeros(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), 256):
        idx = order[s:s + 256]
        enc = tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="longest_first",
                  padding=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            lg = m(**enc).logits.float()
        out[idx] = lg.view(-1).cpu().numpy() if lg.shape[-1] == 1 else lg[:, -1].cpu().numpy()
    return out.tolist()


# ----------------------------------------------------------------------------------------------
# Retrieval orchestrator (CPU, remote)
# ----------------------------------------------------------------------------------------------
def make_queries(splits):
    """Base query = question text (ReClor: question stem only; its passage is given to the model anyway).
    Option-aware queries (R2) = question + ' ' + option text, one per option."""
    qs = []          # list of dicts: key, split, row_id, label (None for base), text
    for sp in splits:
        for r in load_split(sp):
            q = r["question"].strip()
            qs.append({"split": sp, "row_id": r["row_id"], "label": None, "text": q})
            for lab, opt in r["options"].items():
                qs.append({"split": sp, "row_id": r["row_id"], "label": lab, "text": f"{q} {opt}"})
    return qs


def _merge(results, K):
    """results: list of (part, S[nq,K], I[nq,K]) -> per query list of (score, part, local) top-K."""
    import numpy as np
    parts = np.concatenate([np.full(S.shape, p, dtype=np.int64) for p, S, I in results], axis=1)
    S = np.concatenate([S for _, S, _ in results], axis=1)
    I = np.concatenate([I for _, _, I in results], axis=1)
    order = np.argsort(-S, axis=1)[:, :K]
    out = []
    for qi in range(S.shape[0]):
        lst = [(float(S[qi, j]), int(parts[qi, j]), int(I[qi, j])) for j in order[qi] if I[qi, j] >= 0]
        out.append(lst)
    return out


@app.function(volumes=VOLS_RW, cpu=8.0, memory=65536, timeout=12 * 3600)
def retrieve_all(splits: list, corpora: list, K: int = 50, K_opt: int = 20, parts_only: list | None = None,
                 out_name: str = "") -> dict:
    """parts_only/out_name: restrict search to some parts (e.g. med parts 0,1 = textbooks+StatPearls,
    BM25 idf still from the full med corpus) and save under retrieval/{out_name}."""
    import numpy as np
    t0 = time.time()
    ev_vol.reload()
    qs = make_queries(splits)
    base_ix = [i for i, q in enumerate(qs) if q["label"] is None]
    texts = [q["text"] for q in qs]
    print(f"{len(qs)} queries ({len(base_ix)} base)")
    summary = {}
    for corpus in corpora:
        parts = [int(p.stem.split("-")[1]) for p in part_paths(corpus)]
        if parts_only:
            parts = [p for p in parts if p in parts_only]
        # --- BM25 (global idf) ---
        gdf = np.load(EV / "bm25" / corpus / "global_df.npy")
        meta = json.loads((EV / "bm25" / corpus / "global.json").read_text())
        N, avgdl = meta["N"], meta["avgdl"]
        H = hasher()
        Xq = H.transform([t[:4000] for t in texts]).tocsr()
        q_terms, q_idf = [], []
        for qi in range(Xq.shape[0]):
            terms = Xq.indices[Xq.indptr[qi]:Xq.indptr[qi + 1]].astype(np.int64)
            dfv = gdf[terms].astype(np.float64)
            idf = np.log(1 + (N - dfv + 0.5) / (dfv + 0.5))
            q_terms.append(terms.tolist()); q_idf.append(idf.astype(np.float32).tolist())
        bm_res = list(bm25_search_part.starmap([(corpus, p, q_terms, q_idf, avgdl, K) for p in parts]))
        bm = _merge([(p, S, I) for p, (S, I) in zip(parts, bm_res)], K)
        print(f"[{corpus}] bm25 done {time.time() - t0:.0f}s")
        # --- dense ---
        kind = DENSE_ENCODER[corpus]
        Q = encode_queries.remote(kind, texts)
        dn_res = list(dense_search_part.starmap([(corpus, p, Q, K) for p in parts]))
        dn = _merge([(p, S, I) for p, (S, I) in zip(parts, dn_res)], K)
        print(f"[{corpus}] dense done {time.time() - t0:.0f}s")
        # --- fetch texts for every candidate we may use ---
        need = {}
        for qi, q in enumerate(qs):
            k = K if q["label"] is None else K_opt
            for lst in (bm[qi][:k], dn[qi][:k]):
                for _, p, l in lst:
                    need.setdefault(p, set()).add(l)
        fetched = list(fetch_texts.starmap([(corpus, p, sorted(ids)) for p, ids in need.items()]))
        store = {}
        for p, d in zip(need.keys(), fetched):
            for l, v in d.items():
                store[(p, int(l))] = v
        print(f"[{corpus}] fetched {len(store)} passages {time.time() - t0:.0f}s")
        # --- cross-encoder scores (pool = union of bm25/dense candidates; cached by (query, pid)) ---
        pairs, pair_keys = [], []
        for qi, q in enumerate(qs):
            k = K if q["label"] is None else K_opt
            pool = {(p, l) for _, p, l in bm[qi][:k]} | {(p, l) for _, p, l in dn[qi][:k]}
            for pid in sorted(pool):
                d = store[pid]
                pairs.append((q["text"][:3000], doc_string(d["title"], d["text"])[:3000]))
                pair_keys.append((qi, pid))
        chunks = [pairs[s:s + 8000] for s in range(0, len(pairs), 8000)]
        ce_scores = [x for ch in rerank.map([CE_MODEL[corpus]] * len(chunks), chunks) for x in ch]
        ce = {}
        for (qi, pid), s in zip(pair_keys, ce_scores):
            ce.setdefault(qi, {})[pid] = s
        print(f"[{corpus}] reranked {len(pairs)} pairs {time.time() - t0:.0f}s")
        # --- save raw candidate lists per query (for offline config construction) ---
        out = {}
        for qi, q in enumerate(qs):
            k = K if q["label"] is None else K_opt
            key = f"{q['split']}|{q['row_id']}|{q['label'] or ''}"
            out[key] = {"bm25": [[s, p, l] for s, p, l in bm[qi][:k]],
                        "dense": [[s, p, l] for s, p, l in dn[qi][:k]],
                        "ce": [[s, p, l] for (p, l), s in sorted(ce.get(qi, {}).items(), key=lambda x: -x[1])]}
        passages = {f"{p}:{l}": v for (p, l), v in store.items()}
        d = EV / "retrieval" / (out_name or corpus)
        d.mkdir(parents=True, exist_ok=True)
        tag = "_".join(splits)
        (d / f"cands_{tag}.json").write_text(json.dumps(out))
        (d / f"passages_{tag}.json").write_text(json.dumps(passages))
        ev_vol.commit()
        summary[corpus] = {"parts": len(parts), "N": N, "pairs": len(pairs), "s": round(time.time() - t0)}
        print(summary[corpus])
    return summary


# ----------------------------------------------------------------------------------------------
# LLM: R5 background generation and label-only inference with a context file
# ----------------------------------------------------------------------------------------------
def load_qwen(size: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_id = MODELS[size]
    # Same 4-bit config as modal_app/r2r_modal.py::infer
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir="/hf/hub", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant,
                                               trust_remote_code=True, cache_dir="/hf/hub", local_files_only=True)
    mdl.eval()
    return mdl, tok


GEN_SYSTEM = (
    "You are a knowledgeable assistant helping a student prepare to answer a multiple-choice question. "
    "Write concise, factual background knowledge (definitions, mechanisms, principles, formulas) that is "
    "relevant to the question and its options. Do NOT state which option is correct, do NOT answer the "
    "question, and do NOT mention option letters. Use at most 5 short bullet points."
)


def gen_messages(row):
    opts = "\n".join(f"- {t}" for t in row["options"].values())
    psg = f"Passage:\n{row['passage']}\n\n" if row["passage"] else ""
    return [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"{psg}Question:\n{row['question']}\n\nCandidate answers:\n{opts}\n\n"
                                        "Background facts:"}]


@app.function(volumes=VOLS_ALL, gpu=GPU_SMALL, memory=65536, timeout=6 * 3600, retries=1,
              max_containers=GPU_BUDGET["llm"])
def generate_bg(size: str, split: str, row_start: int, row_end: int, max_new_tokens: int = 200) -> dict:
    import torch
    t0 = time.time()
    rows = load_split(split)[row_start:row_end]
    mdl, tok = load_qwen(size)
    tok.padding_side = "left"
    out = {}
    bs = 16
    for s in range(0, len(rows), bs):
        batch = rows[s:s + bs]
        prompts = [tok.apply_chat_template(gen_messages(r), tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(mdl.device)
        with torch.inference_mode():
            g = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, d in zip(batch, dec):
            out[str(r["row_id"])] = d.strip()
        print(f"gen {size}/{split} {s + len(batch)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    d = EV / "gen" / f"qwen{size}" / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f"rows{row_start}-{row_end}.json").write_text(json.dumps(out))
    ev_vol.commit()
    return {"size": size, "split": split, "rows": len(out), "s": round(time.time() - t0)}


def _infer_ctx(size: str, split: str, options: list) -> dict:
    """Label-only answering with reflector_core.build_messages_direct(q, opts, passage, ctx) for each
    context option file /ev/contexts/{split}/{option}.json ({row_id: ctx}); rows with empty ctx skipped."""
    import pandas as pd
    import reflector_core as rc
    t0 = time.time()
    ev_vol.reload()
    rows = load_split(split)
    if all((EV / "runs" / f"qwen2.5-{size}" / split / f"{o}.csv").exists() for o in options):
        return {"size": size, "split": split, "skipped": options}
    mdl, tok = load_qwen(size)
    summ = {}
    for option in options:
        ctxs = json.loads((EV / "contexts" / split / f"{option}.json").read_text())
        out_p = EV / "runs" / f"qwen2.5-{size}" / split / f"{option}.csv"
        if out_p.exists():                       # resumable: finished options are skipped
            summ[option] = "skipped"
            continue
        recs = []
        for row in rows:
            ctx = ctxs.get(str(row["row_id"]), "")
            rec = {"row_id": row["row_id"], "source": row["source"], "gold": row["gold"], "has_ctx": bool(ctx)}
            if ctx:
                o = rc.infer_label_only(rc.build_messages_direct(row["question"], row["options"], row["passage"], ctx),
                                        row["options"], mdl, tok)
                pred = rc.normalize_label(o["predicted_by_probs"])
                rec.update({"pred": pred, "correct": int(pred == row["gold"]), "H": o["entropy_over_labels"],
                            "margin": o["margin"], "top1": o["top1_prob"], "probs": rc.safe_json_dumps(o["label_probs"])})
            recs.append(rec)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(recs)
        df.to_csv(out_p, index=False)
        ev_vol.commit()
        summ[option] = round(float(df["correct"].mean()), 4) if "correct" in df else None
        print(f"{size}/{split}/{option}: acc_on_ctx={summ[option]} n_ctx={int(df.has_ctx.sum())} {time.time()-t0:.0f}s",
              flush=True)
    return {"size": size, "split": split, "acc_on_ctx": summ, "s": round(time.time() - t0)}


@app.function(volumes=VOLS_ALL, gpu=GPU_SMALL, memory=65536, timeout=6 * 3600, retries=1,
              max_containers=GPU_BUDGET["llm"])
def infer_ctx(size: str, split: str, options: list) -> dict:
    return _infer_ctx(size, split, options)


@app.function(volumes=VOLS_ALL, gpu=GPU_BIG, memory=65536, timeout=8 * 3600, retries=1,
              max_containers=GPU_BUDGET["llm"])
def infer_ctx_big(size: str, split: str, options: list) -> dict:
    return _infer_ctx(size, split, options)


# ----------------------------------------------------------------------------------------------
# NLI answer-support gate and contamination scan
# ----------------------------------------------------------------------------------------------
@app.function(gpu=GPU_SMALL, memory=32768, timeout=2 * 3600, retries=2, max_containers=GPU_BUDGET["nli"])
def nli_scores(items: list) -> list:
    """items: (premise, hypothesis) -> P(entailment) with DeBERTa-v3-base-mnli-fever-anli."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    ent = [i for i, l in m.config.id2label.items() if l.lower().startswith("entail")][0]
    out = np.zeros(len(items), dtype=np.float32)
    lens = np.array([len(a) + len(b) for a, b in items])
    order = np.argsort(-lens)
    for s in range(0, len(items), 128):
        idx = order[s:s + 128]
        enc = tok([items[i][0] for i in idx], [items[i][1] for i in idx], truncation="only_first",
                  padding=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            p = torch.softmax(m(**enc).logits.float(), -1)[:, ent]
        out[idx] = p.cpu().numpy()
    return out.tolist()


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


@app.function(volumes=VOLS_RW, cpu=4.0, memory=65536, timeout=3 * 3600, retries=2)
def contam_part(corpus: str, part: int, keys: list) -> dict:
    """Exact-substring scan: normalized question prefix (<=150 chars) inside normalized passages."""
    import pyarrow.parquet as pq
    t = pq.read_table(str(EV / "corpora" / corpus / f"part-{part:05d}.parquet"), columns=["text"])
    big = "\n".join(_norm(x) for x in t.column("text").to_pylist())
    del t
    return {k: big.count(k) for k in keys if k in big}


@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=6 * 3600)
def contam_all(splits: list, corpora: list) -> dict:
    ev_vol.reload()
    keys, owner = [], {}
    for sp in splits:
        for r in load_split(sp):
            k = _norm(r["question"])[:150]
            if len(k) >= 40:
                keys.append(k); owner.setdefault(k, []).append(f"{sp}|{r['row_id']}|{r['source']}")
    keys = sorted(set(keys))
    res = {}
    for corpus in corpora:
        parts = [int(p.stem.split("-")[1]) for p in part_paths(corpus)]
        hits = {}
        for d in contam_part.starmap([(corpus, p, keys) for p in parts]):
            for k, v in d.items():
                hits[k] = hits.get(k, 0) + v
        res[corpus] = {"n_keys": len(keys), "hits": {"; ".join(owner[k]): v for k, v in hits.items()}}
    (EV / "contam.json").write_text(json.dumps(res, indent=1))
    ev_vol.commit()
    return {c: len(v["hits"]) for c, v in res.items()}


# ----------------------------------------------------------------------------------------------
# Orchestrators
# ----------------------------------------------------------------------------------------------
@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=6 * 3600)
def prep_med_range(a: int, b: int) -> list:
    return list(prep_med.map(range(a, b), return_exceptions=True))


@app.function(volumes=VOLS_ALL, cpu=2.0, memory=8192, timeout=12 * 3600)
def orchestrate_prep(corpora: list) -> list:
    calls = []
    if "wiki" in corpora:
        n = len(sorted(Path(WIKI_PARQUET).glob("*.parquet")))
        calls += [prep_wiki.spawn(i) for i in range(n)]
    if "med" in corpora:
        calls += [prep_med.spawn(p) for p in range(2 + N_PUBMED_GROUPS)]
    if "arc" in corpora:
        calls.append(prep_arc.spawn())
    out = []
    for c in calls:
        try:
            out.append(c.get())
        except Exception as e:
            out.append({"error": repr(e)})
        print(out[-1], flush=True)
    return out


@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=12 * 3600)
def orchestrate_index(corpora: list, do_bm25: bool = True, do_dense: bool = True) -> list:
    ev_vol.reload()
    calls = []
    for c in corpora:
        parts = [int(p.stem.split("-")[1]) for p in part_paths(c)]
        if do_bm25:
            calls += [("bm25", c, bm25_part.spawn(c, p)) for p in parts]
        if do_dense:
            calls += [("emb", c, embed_part.spawn(c, p)) for p in parts]
    out = []
    for kind, c, h in calls:
        try:
            out.append(h.get())
        except Exception as e:
            out.append({"kind": kind, "corpus": c, "error": repr(e)})
        print(out[-1], flush=True)
    if do_bm25:
        for c in corpora:
            out.append(bm25_global.remote(c)); print(out[-1], flush=True)
    return out


@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=12 * 3600)
def orchestrate_generate(splits: list, sizes: list, n_shards: int = 4) -> list:
    calls = []
    for size in sizes:
        for sp in splits:
            n = len(load_split(sp))
            edges = [round(i * n / n_shards) for i in range(n_shards + 1)]
            calls += [generate_bg.spawn(size, sp, a, b) for a, b in zip(edges[:-1], edges[1:])]
    out = []
    for c in calls:
        try:
            out.append(c.get())
        except Exception as e:
            out.append({"error": repr(e)})
        print(out[-1], flush=True)
    return out


@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=12 * 3600)
def orchestrate_infer(splits: list, sizes: list, options: list, per_container: int = 1) -> list:
    calls = []
    for size in sizes:
        for sp in splits:
            for s in range(0, len(options), per_container):
                fn = infer_ctx_big if size in BIG_SIZES else infer_ctx
                calls.append(fn.spawn(size, sp, options[s:s + per_container]))
    out = []
    for c in calls:
        try:
            out.append(c.get())
        except Exception as e:
            out.append({"error": repr(e)})
        print(out[-1], flush=True)
    return out


@app.function(volumes=VOLS_RW, cpu=2.0, memory=16384, timeout=6 * 3600)
def orchestrate_nli(splits: list, options: list, n_ctx_passages: int = 3) -> dict:
    """For each (split, option, row, passage-in-context, answer option): P(entail | passage, q_last + option)."""
    ev_vol.reload()
    items, keys = [], []
    for sp in splits:
        rows = {str(r["row_id"]): r for r in load_split(sp)}
        for opt in options:
            P = json.loads((EV / "contexts" / sp / f"{opt}.passages.json").read_text())
            for rid, plist in P.items():
                r = rows[rid]
                sents = re.split(r"(?<=[.?!])\s+", r["question"].strip())
                q_last = sents[-1][-300:] if sents else r["question"][-300:]
                for j, p in enumerate(plist[:n_ctx_passages]):
                    prem = (f"{p['title']}. " if p.get("title") else "") + p["text"]
                    for lab, o in r["options"].items():
                        items.append((prem[:2000], f"{q_last} {o}"[:600]))
                        keys.append((sp, opt, rid, j, lab))
    chunks = [items[s:s + 10000] for s in range(0, len(items), 10000)]
    scores = [x for ch in nli_scores.map(chunks) for x in ch]
    nli_p = EV / "nli.json"
    out = json.loads(nli_p.read_text()) if nli_p.exists() else {}
    for sp in splits:                      # recomputed options replace their old entries
        for opt in options:
            out.get(sp, {}).pop(opt, None)
    for (sp, opt, rid, j, lab), s in zip(keys, scores):
        out.setdefault(sp, {}).setdefault(opt, {}).setdefault(rid, {}).setdefault(lab, []).append(round(s, 5))
    nli_p.write_text(json.dumps(out))
    ev_vol.commit()
    return {"pairs": len(items)}


@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=12 * 3600)
def embed_parts(corpus: str, parts: str) -> list:
    """Embed an explicit list of parts (comma-separated), e.g. in reverse order from a second app."""
    return list(embed_part.starmap([(corpus, int(p)) for p in parts.split(",")], return_exceptions=True))


@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=12 * 3600)
def retrieve_subset(corpus: str, parts: str, out_name: str, splits: str = "validation,test") -> dict:
    return retrieve_all.remote(splits.split(","), [corpus], parts_only=[int(p) for p in parts.split(",")],
                               out_name=out_name)


# ----------------------------------------------------------------------------------------------
# R3 (MedRAG) end to end: missing MedCPT parts -> retrieval (full MedCorp + textbooks/StatPearls subset)
# -> context files -> Qwen inference (4 sizes) -> NLI scores. Every step skips work already on the volume.
# ----------------------------------------------------------------------------------------------
R3_OPTIONS = ["med_bm25_rr", "med_dense", "med_dense_rr", "med_hybrid_rr", "med_hybrid_opt_rr",
              "med_hybrid_rr__L3000", "medtb_dense_rr", "medtb_hybrid_rr"]


@app.function(volumes=VOLS_RW, cpu=4.0, memory=32768, timeout=3600)
def build_contexts_remote(names: list) -> dict:
    """Runs rag_eval/analysis/build_contexts.py against the volume (RES -> /ev)."""
    import os
    import sys
    ev_vol.reload()
    root = Path("/tmp/r2r_root")
    (root / "data").mkdir(parents=True, exist_ok=True)
    if not (root / "data" / "raw").exists():
        os.symlink("/root/data", root / "data" / "raw")
    sys.path.insert(0, "/root/rag_eval/analysis")
    import common
    common.ROOT, common.RES = root, EV
    import build_contexts
    build_contexts.main(names)
    ev_vol.commit()
    return {sp: {n: len(json.loads((EV / "contexts" / sp / f"{n}.json").read_text())) for n in names}
            for sp in ("validation", "test")}


@app.function(volumes=VOLS_RW, cpu=2.0, memory=8192, timeout=24 * 3600)
def orchestrate_r3(splits: list, sizes: list, options: list) -> list:
    log = []

    def note(x):
        print(json.dumps(x, default=str), flush=True)
        log.append(x)

    t0 = time.time()
    ev_vol.reload()
    parts = [int(p.stem.split("-")[1]) for p in part_paths("med")]
    emb = lambda q: EV / "emb" / "med_medcpt" / f"part-{q:05d}.npy"
    missing = [q for q in parts if not emb(q).exists()]
    note({"step": "embed", "missing_parts": missing})
    if missing:
        for r in embed_part.starmap([("med", q) for q in missing], return_exceptions=True):
            note({"embed": repr(r) if isinstance(r, Exception) else r, "t": round(time.time() - t0)})
        ev_vol.reload()
        still = [q for q in parts if not emb(q).exists()]
        if still:
            raise RuntimeError(f"MedCPT embeddings still missing for parts {still}")
    tag = "_".join(splits)
    if not (EV / "retrieval" / "med" / f"cands_{tag}.json").exists():
        note({"step": "retrieve med", "out": retrieve_all.remote(splits, ["med"]), "t": round(time.time() - t0)})
    if not (EV / "retrieval" / "medtb" / f"cands_{tag}.json").exists():
        note({"step": "retrieve medtb", "t": round(time.time() - t0),
              "out": retrieve_all.remote(splits, ["med"], parts_only=[0, 1], out_name="medtb")})
    ev_vol.reload()
    need = [o for o in options if not all((EV / "contexts" / sp / f"{o}.json").exists() for sp in splits)]
    if need:
        note({"step": "contexts", "rows_with_ctx": build_contexts_remote.remote(need), "t": round(time.time() - t0)})
    note({"step": "infer", "out": orchestrate_infer.remote(splits, sizes, options, len(options)),
          "t": round(time.time() - t0)})
    ev_vol.reload()
    nli = json.loads((EV / "nli.json").read_text()) if (EV / "nli.json").exists() else {}
    todo = [o for o in options if not all(o in nli.get(sp, {}) for sp in splits)]
    if todo:
        note({"step": "nli", "out": orchestrate_nli.remote(splits, todo), "t": round(time.time() - t0)})
    note({"step": "done", "t": round(time.time() - t0)})
    return log


@app.local_entrypoint()
def main(stage: str = "prep", corpora: str = "wiki,med,arc", splits: str = "validation,test",
         models: str = "3B,7B,14B,32B", options: str = "", per_container: int = 1, n_shards: int = 4):
    cl = [c for c in corpora.split(",") if c]
    sl = [s for s in splits.split(",") if s]
    ml = [m for m in models.split(",") if m]
    ol = [o for o in options.split(",") if o]
    if stage == "prep":
        print(json.dumps(orchestrate_prep.remote(cl), indent=1))
    elif stage == "index":
        print(json.dumps(orchestrate_index.remote(cl), indent=1))
    elif stage == "bm25":
        print(json.dumps(orchestrate_index.remote(cl, True, False), indent=1))
    elif stage == "dense":
        print(json.dumps(orchestrate_index.remote(cl, False, True), indent=1))
    elif stage == "retrieve":
        print(json.dumps(retrieve_all.remote(sl, cl), indent=1))
    elif stage == "generate":
        print(json.dumps(orchestrate_generate.remote(sl, ml, n_shards), indent=1))
    elif stage == "infer":
        print(json.dumps(orchestrate_infer.remote(sl, ml, ol, per_container), indent=1))
    elif stage == "nli":
        print(json.dumps(orchestrate_nli.remote(sl, ol), indent=1))
    elif stage == "r3":
        print(json.dumps(orchestrate_r3.remote(sl, ml, ol or R3_OPTIONS), indent=1, default=str))
    elif stage == "contam":
        print(json.dumps(contam_all.remote(sl, cl), indent=1))
    else:
        raise ValueError(stage)
