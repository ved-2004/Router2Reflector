"""Generates colab/R2R_RAG_Colab_{3B,7B,14B,32B}.ipynb, one notebook per model size (run: python colab/build_notebook.py).

Each notebook runs ONE Qwen2.5 size in a fresh runtime (loading several sizes in one runtime ran out of GPU memory).
It ports the remaining GPU work of rag_eval/rag_eval_modal.py: R3 medical KB (MedRAG textbooks + StatPearls) contexts
-> Qwen label-only inference (+ NLI gate scores in the 3B notebook). The code mirrors rag_eval_modal.py (same queries,
encoders, reranker, prompt and output format), and each notebook writes r2r_results_<SIZE>.zip, laid out like rag_eval/results/.
"""
import json
from pathlib import Path

GPU_NOTE = {"3B": "any GPU (T4 is fine)", "7B": "any GPU (T4 is fine)", "14B": "T4 (16 GB) or better",
            "32B": "A100 (or L4); a T4 is too small"}
MIN_VRAM_GB = {"3B": 7, "7B": 9, "14B": 13, "32B": 21}


def build(SIZE):
    cells = []

    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": s.strip("\n").splitlines(True)})

    md(f"""
# R2R: replacement RAG evaluation, Qwen2.5-{SIZE}

This notebook runs **one model size ({SIZE})**. The other sizes are in `R2R_RAG_Colab_<size>.ipynb`.

**GPU:** {GPU_NOTE[SIZE]}. Use *Runtime → Change runtime type*. Start from a **fresh runtime**: don't load another model first.

**Input:** `r2r_colab_bundle.zip` in your Google Drive folder `MyDrive/R2R/`.

**Run:** *Runtime → Run all*. If the runtime disconnects, run all again: finished pieces are skipped.

**Output:** `MyDrive/R2R/r2r_results_{SIZE}.zip`. Send this file back.
""")

    code(f"""
# ---- configuration ---------------------------------------------------------------------------
SIZE      = "{SIZE}"
DRIVE_DIR = "/content/drive/MyDrive/R2R"
BUNDLE    = f"{{DRIVE_DIR}}/r2r_colab_bundle.zip"
MED_ZIP   = f"{{DRIVE_DIR}}/r2r_med_contexts.zip"     # optional shared med contexts
OUT       = f"{{DRIVE_DIR}}/colab_results_{{SIZE}}"      # working copy (resume); final zip goes next to it
SPLITS    = ["validation", "test"]
RUN_MED_RETRIEVAL = True        # only runs if the med contexts are not already available
MED_OPTIONS = ["med_bm25_rr", "med_dense", "med_dense_rr", "med_hybrid_rr", "med_hybrid_opt_rr", "med_hybrid_rr__L3000"]
EXTRA_OPTIONS = []
RUN_NLI   = {SIZE == "3B"}       # NLI scores are size-independent: computed once, in the 3B notebook
MIN_VRAM_GB = {MIN_VRAM_GB[SIZE]}
""")

    code(r"""
from google.colab import drive
drive.mount("/content/drive")
import os, subprocess, sys, torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> pick a GPU"
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}  free {free/2**30:.1f} / {total/2**30:.1f} GB")
if free / 2**30 < MIN_VRAM_GB:
    raise RuntimeError(f"{SIZE} needs about {MIN_VRAM_GB} GB of free GPU memory; this GPU has {free/2**30:.1f} GB free. "
                       "Pick a bigger GPU, or Runtime -> Restart session if another model is still loaded.")
os.makedirs(OUT, exist_ok=True)
ROOT = "/content/R2R/Router2Reflector"
if not os.path.exists(ROOT):
    os.makedirs("/content/R2R", exist_ok=True)
    subprocess.run(["unzip", "-q", BUNDLE, "-d", "/content/R2R"], check=True)
if os.path.exists(MED_ZIP):          # shared med contexts (paths relative to rag_eval/results/)
    subprocess.run(["unzip", "-qo", MED_ZIP, "-d", f"{ROOT}/rag_eval/results"], check=True)
    print("using shared med contexts from", MED_ZIP)
# restore anything finished in an earlier session of this notebook
subprocess.run(f"rsync -a --exclude med_corpus {OUT}/ {ROOT}/rag_eval/results/", shell=True)
os.chdir(ROOT)
sys.path[:0] = [f"{ROOT}/src", f"{ROOT}/rag_eval", f"{ROOT}/rag_eval/analysis"]
""")

    code(r"""
!pip -q install "transformers==4.46.3" "accelerate==1.1.1" "bitsandbytes>=0.45.0" "bm25s==0.2.5" "PyStemmer==2.2.0.3"
""")

    code(r"""
import json, re, time, gc, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch

RES = Path(ROOT) / "rag_eval" / "results"
DATA = Path(ROOT) / "data" / "raw"
SPLIT_FILES = {"train": "Train.csv", "validation": "Validation.csv", "test": "Test.csv"}
MODELS = {s: f"Qwen/Qwen2.5-{s}-Instruct" for s in ["3B", "7B", "14B", "32B"]}
DEV = "cuda"

def save_to_drive(rel):
    # copy rag_eval/results/<rel> (file or dir) into the Drive mirror
    src, dst = RES / rel, Path(OUT) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

def load_split(split):
    # identical to rag_eval_modal.load_split
    import ast
    from reflector_core import get_passage_for_row, normalize_label
    df = pd.read_csv(DATA / SPLIT_FILES[split]).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        labels, texts = ast.literal_eval(r["choice_labels"]), ast.literal_eval(r["choice_texts"])
        rows.append({"row_id": int(i), "source": str(r["source"]), "question": str(r["question"]),
                     "options": dict(zip(labels, texts)),
                     "passage": get_passage_for_row(str(r["source"]), r.get("context")),
                     "gold": normalize_label(str(r["gold_label"]))})
    return rows

def make_queries(splits):
    # base query = question; option-aware query = question + ' ' + option (rag_eval_modal.make_queries)
    qs = []
    for sp in splits:
        for r in load_split(sp):
            q = r["question"].strip()
            qs.append({"split": sp, "row_id": r["row_id"], "label": None, "text": q})
            for lab, opt in r["options"].items():
                qs.append({"split": sp, "row_id": r["row_id"], "label": lab, "text": f"{q} {opt}"})
    return qs

def doc_string(title, text):
    title = (title or "").strip()
    return f"{title}. {text}" if title else text
print("ok")
    """)

    md(r"""
## 1. R3 medical knowledge base: MedRAG textbooks + StatPearls

* part 0 = `MedRAG/textbooks` (HF), part 1 = StatPearls (NCBI tarball, chunked with MedRAG's own
  chunker, `rag_eval/statpearls_chunk.py`). PubMed (23.9M snippets) is left out on Colab.
* The corpus parquet is cached in Drive (`OUT/med_corpus/`), so it is built only once.
    """)

    code(r"""
MED_DONE = (RES / "retrieval" / "med" / "cands_validation_test.json").exists() or all(
    (RES / "contexts" / sp / f"{o}.json").exists() for sp in SPLITS for o in MED_OPTIONS)
print("med retrieval already done:", MED_DONE)
if RUN_MED_RETRIEVAL and not MED_DONE:
    import pyarrow as pa, pyarrow.parquet as pq, tarfile, requests
    from huggingface_hub import list_repo_files, hf_hub_download
    cdir = Path(OUT) / "med_corpus"; cdir.mkdir(parents=True, exist_ok=True)
    if not (cdir / "part-00000.parquet").exists():
        T, X, S = [], [], []
        for fn in sorted(f for f in list_repo_files("MedRAG/textbooks", repo_type="dataset") if f.endswith(".jsonl")):
            for line in open(hf_hub_download("MedRAG/textbooks", fn, repo_type="dataset")):
                if line.strip():
                    r = json.loads(line); T.append(r["title"]); X.append(r["content"]); S.append("textbooks")
        pq.write_table(pa.table({"title": T, "text": X, "src": S}), str(cdir / "part-00000.parquet"), compression="zstd")
        print("textbooks", len(X))
    if not (cdir / "part-00001.parquet").exists():
        os.makedirs("/content/sp", exist_ok=True)
        url = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
        with requests.get(url, stream=True, timeout=600) as r, open("/content/sp/sp.tgz", "wb") as fh:
            r.raise_for_status()
            for ch in r.iter_content(1 << 22): fh.write(ch)
        with tarfile.open("/content/sp/sp.tgz") as tf: tf.extractall("/content/sp/x", filter="data")
        from statpearls_chunk import extract
        T, X, S = [], [], []
        for fp in sorted(Path("/content/sp/x").rglob("*.nxml")):
            try:
                for s in extract(str(fp)):
                    r = json.loads(s); T.append(r["title"]); X.append(r["content"]); S.append("statpearls")
            except Exception as e:
                print("skip", fp.name, e)
        pq.write_table(pa.table({"title": T, "text": X, "src": S}), str(cdir / "part-00001.parquet"), compression="zstd")
        print("statpearls", len(X))
    parts = [pq.read_table(str(cdir / f"part-0000{p}.parquet")).to_pydict() for p in (0, 1)]
    PID = [(p, l) for p in (0, 1) for l in range(len(parts[p]["text"]))]   # global id -> (part, local)
    TITLES = parts[0]["title"] + parts[1]["title"]; TEXTS = parts[0]["text"] + parts[1]["text"]
    SRCS = parts[0]["src"] + parts[1]["src"]
    print("med docs:", len(TEXTS))
    """)

    md(r"""
### Retrieval: BM25, MedCPT dense and MedCPT cross-encoder (K = 50 base, 20 per option)
These match `rag_eval_modal.retrieve_all`. The one difference is that BM25 here is `bm25s` (Lucene idf, k1=1.5,
b=0.75, the same English stopwords and Snowball stemmer) with an exact vocabulary instead of the 2^24 hashed one.
    """)

    code(r"""
if RUN_MED_RETRIEVAL and not MED_DONE:
    import bm25s, Stemmer
    K, K_OPT = 50, 20
    qs = make_queries(SPLITS); texts = [q["text"] for q in qs]
    print(len(qs), "queries")
    stem = Stemmer.Stemmer("english")
    t0 = time.time()
    bm = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    bm.index(bm25s.tokenize([doc_string(a, b) for a, b in zip(TITLES, TEXTS)], stopwords="en", stemmer=stem, show_progress=False), show_progress=False)
    qtok = bm25s.tokenize([t[:4000] for t in texts], stopwords="en", stemmer=stem, return_ids=False, show_progress=False)
    qtok = [[w for w in q if w in bm.vocab_dict] for q in qtok]
    empty = [i for i, q in enumerate(qtok) if not q]
    any_tok = next(iter(bm.vocab_dict))
    BI, BS = bm.retrieve([q or [any_tok] for q in qtok], k=K, show_progress=False, n_threads=os.cpu_count())
    BS[empty] = 0.0                                    # no in-vocabulary terms -> no BM25 candidates
    print(f"bm25 {time.time()-t0:.0f}s")

    from transformers import AutoModel, AutoTokenizer
    def medcpt_encode(name, items, bs=256, maxlen=512):
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
        lens = np.array([len(x[0]) + len(x[1]) if isinstance(x, (list, tuple)) else len(x) for x in items])
        order = np.argsort(-lens); out = np.zeros((len(items), 768), dtype=np.float16)
        for s in range(0, len(items), bs):
            idx = order[s:s + bs]
            enc = tok([items[i] for i in idx], truncation=True, padding=True, max_length=maxlen, return_tensors="pt").to(DEV)
            with torch.inference_mode():
                out[idx] = m(**enc).last_hidden_state[:, 0, :].float().cpu().numpy().astype(np.float16)
        del m; torch.cuda.empty_cache()
        return out
    emb_path = Path(OUT) / "med_corpus" / "medcpt_emb.npy"
    if emb_path.exists():
        E = np.load(emb_path)
    else:
        E = medcpt_encode("ncbi/MedCPT-Article-Encoder", [[a, b] for a, b in zip(TITLES, TEXTS)])
        np.save(emb_path, E)
    print(f"doc emb {E.shape} {time.time()-t0:.0f}s")
    Q = medcpt_encode("ncbi/MedCPT-Query-Encoder", texts, bs=128)
    Et = torch.from_numpy(E).to(DEV); Qt = torch.from_numpy(Q).to(DEV)
    DS, DI = [], []
    for s in range(0, len(Qt), 512):
        v, i = torch.topk((Qt[s:s+512] @ Et.T).float(), K, dim=1); DS.append(v.cpu().numpy()); DI.append(i.cpu().numpy())
    DS, DI = np.concatenate(DS), np.concatenate(DI); del Et, Qt; torch.cuda.empty_cache()
    print(f"dense {time.time()-t0:.0f}s")

    from transformers import AutoModelForSequenceClassification
    ce_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    ce_m = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder", torch_dtype=torch.float16).to(DEV).eval()
    pairs, keys = [], []
    for qi, q in enumerate(qs):
        k = K if q["label"] is None else K_OPT
        pool = {int(g) for g, s in zip(BI[qi][:k], BS[qi][:k]) if s > 0} | {int(g) for g in DI[qi][:k]}
        for g in sorted(pool):
            pairs.append((q["text"][:3000], doc_string(TITLES[g], TEXTS[g])[:3000])); keys.append((qi, g))
    print(len(pairs), "CE pairs")
    lens = np.array([len(a) + len(b) for a, b in pairs]); order = np.argsort(-lens); ce_s = np.zeros(len(pairs), np.float32)
    for s in range(0, len(pairs), 256):
        idx = order[s:s + 256]
        enc = ce_tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="longest_first",
                     padding=True, max_length=512, return_tensors="pt").to(DEV)
        with torch.inference_mode():
            lg = ce_m(**enc).logits.float()
        ce_s[idx] = (lg.view(-1) if lg.shape[-1] == 1 else lg[:, -1]).cpu().numpy()
    del ce_m; torch.cuda.empty_cache()
    print(f"rerank {time.time()-t0:.0f}s")

    ce = {}
    for (qi, g), s in zip(keys, ce_s): ce.setdefault(qi, {})[g] = float(s)
    trip = lambda s, g: [float(s), PID[g][0], PID[g][1]]
    out, used = {}, set()
    for qi, q in enumerate(qs):
        k = K if q["label"] is None else K_OPT
        b = [trip(s, int(g)) for g, s in zip(BI[qi][:k], BS[qi][:k]) if s > 0]
        d = [trip(s, int(g)) for g, s in zip(DI[qi][:k], DS[qi][:k])]
        c = [trip(s, g) for g, s in sorted(ce.get(qi, {}).items(), key=lambda x: -x[1])]
        used |= {g for g in ce.get(qi, {})}
        out[f"{q['split']}|{q['row_id']}|{q['label'] or ''}"] = {"bm25": b, "dense": d, "ce": c}
    passages = {f"{PID[g][0]}:{PID[g][1]}": {"title": TITLES[g], "text": TEXTS[g], "src": SRCS[g]} for g in used}
    d = RES / "retrieval" / "med"; d.mkdir(parents=True, exist_ok=True)
    (d / "cands_validation_test.json").write_text(json.dumps(out))
    (d / "passages_validation_test.json").write_text(json.dumps(passages))
    save_to_drive("retrieval/med")
    print("saved med retrieval", time.time() - t0)
    """)


    md("## 2. Build the med context files (skipped if they already exist)")

    code(r"""
need = [o for o in MED_OPTIONS if not all((RES / "contexts" / sp / f"{o}.json").exists() for sp in SPLITS)]
BUILT_MED_HERE = bool(need)
if need:
    !python rag_eval/analysis/build_contexts.py {" ".join(need)}
    save_to_drive("contexts")
OPTIONS = MED_OPTIONS + EXTRA_OPTIONS
missing = [o for o in OPTIONS for sp in SPLITS if not (RES / "contexts" / sp / f"{o}.json").exists()]
assert not missing, f"missing context files: {missing}"
print("options to run:", OPTIONS)
""")

    md(f"""
## 3. Qwen2.5-{SIZE} label-only inference with each context (resumable)
This is the same code as `rag_eval_modal.infer_ctx`: `reflector_core.build_messages_direct(q, opts, passage, ctx)`
followed by `infer_label_only`. The whole model is placed on the GPU (`device_map={{"": 0}}`), so it never offloads to the CPU.
""")

    code(r"""
import gc
import reflector_core as rc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
gc.collect(); torch.cuda.empty_cache()

def load_qwen(size):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODELS[size], trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(MODELS[size], device_map={"": 0}, quantization_config=quant,
                                               low_cpu_mem_usage=True, trust_remote_code=True).eval()
    return mdl, tok

ROWS = {sp: load_split(sp) for sp in SPLITS}
todo = [(sp, o) for sp in SPLITS for o in OPTIONS
        if not (Path(OUT) / "runs" / f"qwen2.5-{SIZE}" / sp / f"{o}.csv").exists()]
print("to run:", todo)
if todo:
    t0 = time.time(); mdl, tok = load_qwen(SIZE)
    print(f"{SIZE} loaded {time.time()-t0:.0f}s, GPU mem {torch.cuda.memory_allocated()/2**30:.1f} GB")
    for sp, option in todo:
        ctxs = json.loads((RES / "contexts" / sp / f"{option}.json").read_text())
        recs = []
        for row in ROWS[sp]:
            ctx = ctxs.get(str(row["row_id"]), "")
            rec = {"row_id": row["row_id"], "source": row["source"], "gold": row["gold"], "has_ctx": bool(ctx)}
            if ctx:
                o = rc.infer_label_only(rc.build_messages_direct(row["question"], row["options"], row["passage"], ctx),
                                        row["options"], mdl, tok)
                pred = rc.normalize_label(o["predicted_by_probs"])
                rec.update({"pred": pred, "correct": int(pred == row["gold"]), "H": o["entropy_over_labels"],
                            "margin": o["margin"], "top1": o["top1_prob"], "probs": rc.safe_json_dumps(o["label_probs"])})
            recs.append(rec)
        p = RES / "runs" / f"qwen2.5-{SIZE}" / sp / f"{option}.csv"; p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(recs); df.to_csv(p, index=False); save_to_drive(f"runs/qwen2.5-{SIZE}/{sp}/{option}.csv")
        acc = df["correct"].mean() if "correct" in df else float("nan")
        print(f"{SIZE}/{sp}/{option}: acc_on_ctx={acc:.3f} n_ctx={int(df.has_ctx.sum())} {time.time()-t0:.0f}s")
    del mdl, tok; gc.collect(); torch.cuda.empty_cache()
""")

    md(r"""
## 4. NLI answer-support scores for the new options (optional)
This is the same as `rag_eval_modal.orchestrate_nli`: P(entailment | passage, last question sentence + option) from
DeBERTa-v3-base-mnli-fever-anli, using the top 3 passages in each context. The new scores are merged into `nli.json`.
    """)

    code(r"""
if RUN_NLI and OPTIONS:
    from transformers import AutoModelForSequenceClassification
    nli_path = RES / "nli.json"
    NLI = json.loads(nli_path.read_text()) if nli_path.exists() else {}
    todo = [(sp, o) for sp in SPLITS for o in OPTIONS if o not in NLI.get(sp, {})]
    if todo:
        name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ntok = AutoTokenizer.from_pretrained(name)
        nm = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
        ent = [i for i, l in nm.config.id2label.items() if l.lower().startswith("entail")][0]
        items, keys = [], []
        for sp, opt in todo:
            rows = {str(r["row_id"]): r for r in ROWS[sp]}
            P = json.loads((RES / "contexts" / sp / f"{opt}.passages.json").read_text())
            for rid, plist in P.items():
                r = rows[rid]
                sents = re.split(r"(?<=[.?!])\s+", r["question"].strip())
                q_last = sents[-1][-300:] if sents else r["question"][-300:]
                for j, p in enumerate(plist[:3]):
                    prem = (f"{p['title']}. " if p.get("title") else "") + p["text"]
                    for lab, o in r["options"].items():
                        items.append((prem[:2000], f"{q_last} {o}"[:600])); keys.append((sp, opt, rid, j, lab))
        lens = np.array([len(a) + len(b) for a, b in items]); order = np.argsort(-lens); sc = np.zeros(len(items), np.float32)
        for s in range(0, len(items), 128):
            idx = order[s:s + 128]
            enc = ntok([items[i][0] for i in idx], [items[i][1] for i in idx], truncation="only_first",
                       padding=True, max_length=512, return_tensors="pt").to(DEV)
            with torch.inference_mode():
                sc[idx] = torch.softmax(nm(**enc).logits.float(), -1)[:, ent].cpu().numpy()
        for (sp, opt, rid, j, lab), s in zip(keys, sc):
            NLI.setdefault(sp, {}).setdefault(opt, {}).setdefault(rid, {}).setdefault(lab, []).append(round(float(s), 5))
        nli_path.write_text(json.dumps(NLI)); save_to_drive("nli.json")
        del nm; torch.cuda.empty_cache()
        print("nli pairs:", len(items))
    """)


    md("## 5. Quick check: accuracy vs the direct answer (always inject when a context exists)")

    code(r"""
rows = []
for sp in SPLITS:
    d = pd.read_csv(Path(ROOT) / "outputs" / "pipeline_runs" / f"qwen2.5-{SIZE}" / f"{sp}.csv")[["row_id", "direct_correct"]]
    for o in OPTIONS:
        r = pd.read_csv(RES / "runs" / f"qwen2.5-{SIZE}" / sp / f"{o}.csv").merge(d, on="row_id")
        final = np.where(r["has_ctx"], r.get("correct", np.nan), r["direct_correct"]).astype(float)
        rows.append({"split": sp, "option": o, "direct": round(r.direct_correct.mean() * 100, 1),
                     "with_ctx": round(np.nanmean(final) * 100, 1),
                     "delta_pp": round((np.nanmean(final) - r.direct_correct.mean()) * 100, 1),
                     "injected": int(r.has_ctx.sum())})
print(pd.DataFrame(rows).pivot(index="option", columns="split", values="delta_pp"))
""")

    md(f"## 6. Package the results: `MyDrive/R2R/r2r_results_{SIZE}.zip`")

    code(r"""
import zipfile
zpath = f"{DRIVE_DIR}/r2r_results_{SIZE}.zip"
files = sorted((RES / "runs" / f"qwen2.5-{SIZE}").rglob("*.csv"))
if RUN_NLI and (RES / "nli.json").exists():
    files.append(RES / "nli.json")
if BUILT_MED_HERE or not MED_DONE:     # med retrieval was produced in this notebook
    files += sorted((RES / "retrieval" / "med").glob("*.json")) + sorted((RES / "contexts").glob("*/med_*.json"))
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(RES))
n_runs = sum("runs/" in str(f) for f in files)
print(f"wrote {zpath}: {len(files)} files ({n_runs} run CSVs, expected {len(OPTIONS) * len(SPLITS)})")
""")

    md(f"""
## Done
Send **`MyDrive/R2R/r2r_results_{SIZE}.zip`** back. Locally, unzip it into `Router2Reflector/rag_eval/results/`
(`unzip -o r2r_results_{SIZE}.zip -d rag_eval/results`).
""")

    return {"cells": cells, "metadata": {"accelerator": "GPU",
                                         "colab": {"provenance": [], "gpuType": "A100" if SIZE == "32B" else "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for s in ["3B", "7B", "14B", "32B"]:
        p = here / f"R2R_RAG_Colab_{s}.ipynb"
        p.write_text(json.dumps(build(s), indent=1))
        print("wrote", p)
