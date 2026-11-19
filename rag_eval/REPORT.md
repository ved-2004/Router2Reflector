# R2R replacement-RAG evaluation: report

**Status (2026-11-18): complete** on Validation and Test for all 4 sizes: Stage 1 and Stage 2 for R0–R6, every gate, NLI scores and the contamination checks.
- R3 (MedRAG) was finished on 2026-11-18 on Modal with budget GPUs (L4 / L40S, no H100s). See "R3 run" under Costs.
- Only the optional Train split is not run.

**How to read the numbers.**
- Every number is on Test (n=300, 50 per source) unless it says otherwise. Δ is against the existing `direct_*` pass.
- CIs are paired-bootstrap 95% (2000 resamples).
- 1 pp = 3 questions, and 1 question on one source = 2 pp.
- About 19 options × 5 gates × 4 sizes are compared. A single CI that excludes 0 by less than ~1 pp is noise.
- The 2026-11-18 re-analysis reproduces every earlier point estimate, injection rate and gate threshold exactly. CIs moved by ≤0.7 pp, because the bootstrap's random stream now covers more options.

**Option codes.**

| code | option |
|---|---|
| R0 | current RAG: BM25 over Wikipedia leads, filtered |
| R1 | full-article Wikipedia chunks |
| R2 | option-aware retrieval (question + option as query) |
| R3 | MedRAG medical corpus. `med_*` = full MedCorp (textbooks + StatPearls + PubMed, 24.4M snippets); `medtb_*` = textbooks + StatPearls only (510k) |
| R4 | ARC science corpus |
| R5 | generate-then-read |
| R6 | oracle passage |

## TL;DR
1. **No tested retrieval option gives a reliable gain at any size. That now includes R3.**
   - Always-on means range from −2.3 pp (R1 dense) to −6.4 pp (R2).
   - R3 lies between −2.4 pp (MedCorp MedCPT + CE) and −4.3 pp (MedCorp hybrid, 3000 chars).
   - Every option breaks about 1.5–3.5× as many answers as it fixes.
2. **R3 does not fix MedQA, the case it was meant for.**
   - On Test, every R3 option hurts MedQA (mean −1.5 to −12 pp).
   - The best is textbooks+StatPearls hybrid at −1.5 pp, which is no better than Wikipedia dense (−4.5) within noise.
   - On Validation, textbooks+StatPearls dense+CE looked like a gain on MedQA (+8 pp mean, positive at every size). It **reversed** on Test (−5.5 pp, negative at every size).
   - The per-source gate, fitted on Validation, turns on MedQA retrieval for most sizes and loses on Test every time. With 50 MedQA questions per split, that Validation gain was noise.
3. **Retrieval quality is not the only problem.**
   - R1 dense roughly doubles answer recall over R0 (Test recall@3 14.4 vs 8.9).
   - R3 recall is lower than R1 overall and no better on MedQA: recall@3 on MedQA is 1–6% vs 5% for R1. MedQA answers are clinical phrases that rarely appear verbatim.
   - When the context holds the gold answer and no distractor, the models use it: R1 +11 fixed / −1 broken, R3 MedCorp dense +5 / −1.
   - That covers only about 10% of questions. Where the gold answer is absent, the context breaks about 2× more than it fixes: R1 +56 / −103, R3 MedCorp dense +54 / −92.
4. **R6 (oracle SciQ gold passage)** gives +4 to +6 pp on SciQ at every size, with 0 broken.
5. **R5 generate-then-read** (32B writes the background) is within noise: +0.3 / +1.0 / −1.7 / −3.0. Self-generated background is worse.
6. **Gates** fitted on Validation (direct_H, NLI support, per-source) bring Test Δ to about 0, mostly by retrieving rarely. Routing AQUA-RAT and ReClor off roughly halves the loss. This holds for R3 too: H-gate means are −1.0 to −0.2 and NLI-gate means −1.6 to −0.2.
7. **Recommendation:**
   - Don't adopt any tested option always-on, and drop R0.
   - Don't add a medical corpus. It costs a 24M-snippet index and still loses on MedQA.
   - The least-bad option is still R1 dense behind a gate, with AQUA-RAT/ReClor routed off. Expect about 0 pp.
   - For the combined pipeline, claim detection/abstention and cost, not accuracy gains from retrieval.

## Ranked options (Test Δ pp, always-on)

| option | 3B (71.3) | 7B (75.3) | 14B (84.0) | 32B (88.7) | mean |
|---|---|---|---|---|---|
| R6 oracle SciQ gold passage | +0.7 [0.0,+1.7] | +1.0 [0.0,+2.3] | +1.0 [0.0,+2.3] | +0.7 [0.0,+1.7] | +0.8 |
| R5 generate-then-read (32B writes) | +0.3 [−4.3,+5.0] | +1.0 [−2.3,+4.3] | −1.7 [−4.7,+1.3] | −3.0 [−6.0,−0.3] | −0.8 |
| R1 wiki dense, 3000 chars / 5 passages | −4.0 [−8.3,0.0] | −1.0 [−4.3,+2.7] | −1.7 [−4.7,+1.3] | −1.7 [−4.0,+0.7] | −2.1 |
| R5 self-generated | −4.7 [−9.0,−0.3] | 0.0 [−3.0,+3.3] | −0.7 [−4.3,+3.0] | −3.0 [−6.0,−0.3] | −2.1 |
| **R1 wiki full-article chunks, dense bge-base** | −3.3 [−7.3,+0.7] | −1.3 [−4.3,+1.7] | −2.3 [−5.7,+0.7] | −2.3 [−5.0,+0.3] | −2.3 |
| R3 MedCorp MedCPT + MedCPT-CE | −4.7 [−8.3,−1.0] | −1.7 [−5.0,+2.0] | −1.7 [−4.7,+1.3] | −1.7 [−4.0,+0.3] | −2.4 |
| R3 textbooks+StatPearls hybrid + CE | −2.7 [−6.3,+0.7] | −1.7 [−4.7,+1.0] | −3.0 [−6.7,0.0] | −3.0 [−5.7,−0.3] | −2.6 |
| R3 MedCorp MedCPT dense | −4.7 [−8.7,−0.7] | −2.0 [−5.3,+1.3] | −1.7 [−4.3,+1.0] | −3.3 [−5.7,−1.3] | −2.9 |
| R4 ARC Corpus dense (top-5 sentences) | −4.3 [−7.7,−1.3] | −2.3 [−5.3,+0.3] | −1.7 [−5.0,+1.3] | −4.0 [−6.7,−1.3] | −3.1 |
| R3 textbooks+StatPearls MedCPT + CE | −5.0 [−8.7,−1.3] | −2.7 [−5.7,+0.7] | −3.0 [−6.3,+0.3] | −2.3 [−5.0,0.0] | −3.2 |
| R3 MedCorp BM25 + MedCPT-CE | −5.0 [−9.3,−0.7] | −1.7 [−5.3,+2.0] | −3.3 [−6.7,−0.3] | −3.3 [−6.0,−1.0] | −3.3 |
| R3 MedCorp hybrid (BM25 + MedCPT) + CE | −5.3 [−9.3,−1.3] | −2.3 [−6.0,+1.3] | −3.0 [−6.3,0.0] | −2.7 [−5.0,−0.7] | −3.3 |
| R0 current (BM25 wiki leads, filtered) | −3.7 [−7.3,−0.3] | −2.0 [−4.7,+1.0] | −4.0 [−7.3,−0.7] | −3.7 [−6.3,−1.0] | −3.3 |
| R1 dense + bge-reranker-base | −3.0 [−7.0,+0.7] | −2.7 [−6.0,+0.7] | −4.7 [−8.3,−1.3] | −4.0 [−7.0,−1.0] | −3.6 |
| R2+R3 MedCorp option-aware + CE | −6.3 [−10.7,−1.7] | −2.3 [−6.0,+1.3] | −4.0 [−7.7,−0.7] | −4.0 [−7.0,−1.3] | −4.2 |
| R4 ARC option-aware hybrid + reranker | −6.0 [−10.3,−1.7] | −3.0 [−6.3,+0.3] | −3.7 [−7.0,−0.3] | −4.0 [−6.7,−1.3] | −4.2 |
| R3 MedCorp hybrid + CE, 3000 chars / 5 passages | −7.3 [−12.0,−3.0] | −2.7 [−6.0,+0.7] | −3.7 [−7.0,−0.3] | −3.7 [−6.0,−1.7] | −4.3 |
| R2 wiki option-aware hybrid + reranker | −10.3 [−15.0,−5.7] | −6.7 [−10.7,−3.0] | −4.3 [−7.7,−1.0] | −4.3 [−7.3,−1.7] | −6.4 |

**Fixed/broken counts** (Test, always-on, sizes 3B / 7B / 14B / 32B):
- R0: +9/−20, +7/−13, +6/−18, +3/−14
- R1 dense: +13/−23, +9/−13, +9/−16, +6/−13
- R2: +12/−43 for 3B
- R3 MedCorp MedCPT + CE: +10/−24, +12/−17, +8/−13, +3/−8
- R3 textbooks+StatPearls hybrid + CE: +11/−19, +6/−11, +9/−18, +4/−13
- R5-32B: +26/−25, +14/−11, +8/−13, +5/−14
- R6: +2 to +3 fixed, 0 broken at every size

**R5 cross-generator matrix.** Each row is the model that writes the background; the four values are Δ for the reader sizes 3B / 7B / 14B / 32B.
- 3B writes: −4.7 / −4.3 / −6.3 / −9.0
- 7B writes: +1.7 / 0.0 / −1.7 / −3.7
- 14B writes: +2.0 / +3.3 [−0.3, +7.0] / −0.7 / −2.3
- 32B writes: +0.3 / +1.0 / −1.7 / −3.0

The 14B→7B +3.3 is not significant. It comes mostly from AQUA-RAT (+16), where the generated text works like a chain-of-thought hint.

## Gates (thresholds fitted on Validation per size and option, applied to Test)
- **route:** never inject for AQUA-RAT or ReClor. Roughly halves the loss but stays ≤ 0.
  - R1 dense: −2.0 / −0.7 / −1.0 / −1.0.
  - R3: means −4.3 to −1.8. The best is textbooks+StatPearls hybrid: −1.7 / −1.7 / −2.7 / −1.3.
- **H:** route, plus inject only if direct_H ≥ τ (τ from a Validation quantile grid).
  - Every retrieval option ends up between −1.0 and 0.0 mean, at 2–17% injection.
  - R1 dense: −0.7 / 0.0 / +0.7 / 0.0. R3 options: −1.0 to −0.2 mean, at 4–10% injection.
- **nli:** route, plus inject only if the top option's P(entail) ≥ τ and its gap to the next option ≥ δ.
  - Model: DeBERTa-v3-base-mnli-fever-anli. Premise: each context passage. Hypothesis: last question sentence + option.
  - Retrieval options end up between −1.6 and −0.1, at 6–28% injection; R5-32B is 0.0 mean.
  - Among the R3 options, only MedCorp hybrid at 3000 chars loses clearly (−6.0 for 3B). For 3B the fit chose τ = 0, δ = 0, which is just the route gate at 67% injection, because that won on Validation by 2 questions.
- **src:** per-source on/off, chosen on Validation. Noisy; R6 is the only consistent positive (+0.7).
  - For R3 it turns MedQA on in 19 of 32 (size, option) cells (all 8 options for 3B). Every one of those cells is ≤ 0 on Test (−3.7 to 0.0).

Every fitted gate ends up close to "inject almost never". There is no reliable subset of questions where retrieval helps.

## Per-source (Test, always-on, 3B / 7B / 14B / 32B)
- **MedQA:**
  - R0: −6 / −4 / −4 / −10
  - R1 dense: 0 / −4 / −4 / −10
  - R1 + reranker: −2 / −10 / −12 / −12
  - R2: −26 / −16 / −12 / −16
  - R3 MedCorp MedCPT: −8 / −4 / 0 / −10
  - R3 MedCorp MedCPT + CE: −4 / −6 / −6 / −8
  - R3 MedCorp hybrid + CE: −12 / 0 / −12 / −8
  - R3 MedCorp option-aware + CE: −8 / −12 / −14 / −14
  - R3 textbooks+StatPearls MedCPT + CE: −6 / −6 / −4 / −6. On **Validation** it was +12 / +8 / +8 / +4.
  - R3 textbooks+StatPearls hybrid + CE: 0 / 0 / −2 / −4 (the best R3 option on Test MedQA). On Validation it was +6 / +4 / +2 / −2.
  - R4: −10 / −8 / 0 / −12
  - R5-32B: +2 / +6 / 0 / −8. On Validation it was +26 / +14 / +6 / +6, so it does not replicate either.
- **SciQ:**
  - R1 dense: −6 / +4 / +4 / +4. Not replicated on Validation (−2 / +2 / 0 / 0).
  - Oracle: +4 / +6 / +6 / +4.
  - R3: every option is between −14 and +4.
- **MMLU:** every option is between −8 and +6. The +6 is 3 questions.
- **ARC-Easy:** every option is between 0 and −12.
- **AQUA-RAT:** R3 MedCorp options are between −8 and +10. That source is routed off by every gate.

The full grid is in `results/tables.md` and `results/stage2_per_source.csv`.

## Stage 1: answer recall (Test, n=270)

| config | recall@1 | recall@3 | recall@5 | discrim@3 |
|---|---|---|---|---|
| R0 | 4.8 | 8.9 | 8.9 | 7.8 |
| R1 BM25 | 5.9 | 10.4 | 12.6 | 7.8 |
| **R1 dense** | 7.4 | 14.4 | 19.3 | 10.7 |
| R1 dense + reranker | 8.9 | 15.2 | 18.9 | 9.6 |
| R2 option-aware hybrid + reranker | 13.7 | 23.3 | 25.6 | 13.0 |
| R3 MedCorp BM25 | 4.4 | 7.8 | 11.1 | 5.6 |
| R3 MedCorp MedCPT | 4.1 | 8.1 | 9.3 | 5.9 |
| R3 MedCorp MedCPT + CE | 4.4 | 9.3 | 11.9 | 5.9 |
| R3 MedCorp hybrid + CE | 4.4 | 10.0 | 12.6 | 8.1 |
| R3 MedCorp option-aware + CE | 13.7 | 21.1 | 24.4 | 11.9 |
| R3 textbooks+StatPearls MedCPT + CE | 3.7 | 7.0 | 8.9 | 5.2 |
| R3 textbooks+StatPearls hybrid + CE | 4.8 | 8.1 | 11.1 | 5.9 |
| R4 ARC dense | 6.7 | 11.1 | 12.6 | 9.6 |

- Option-aware recall (R2, and R2+R3) is inflated, because the option text is part of the query. Its discriminativeness barely rises, because it also retrieves passages that support distractors.
- **On rows the model gets wrong** (Val+Test):
  - R1 dense recall@3 is 5.3% for 3B and 6.5% for 32B. R0 is 2.0% and 4.8%.
  - R3 MedCorp hybrid + CE is 3.3% and 4.8%. R3 textbooks+StatPearls hybrid + CE is 2.7% and 6.5%.
- **Recall@3 by source** (Val+Test):
  - R1 dense: SciQ 58, ARC-Easy 20, MedQA 5, MMLU 2, AQUA 4, ReClor 0.
  - R0: SciQ 35, ARC-Easy 5, MedQA 3, MMLU 0, AQUA 2, ReClor 0.
  - R3 MedCorp hybrid + CE: SciQ 39, ARC-Easy 10, MedQA 3, MMLU 4.
  - R3 textbooks+StatPearls hybrid + CE: SciQ 38, ARC-Easy 6, MedQA 6, MMLU 1.
  - Even on MedQA, the medical corpus does not retrieve the gold answer text more often than Wikipedia.

**Fixed/broken by whether the context holds the answer** (`results/answer_bearing.csv`; Val+Test pooled, 4 sizes, injected rows only):

| option | context answer-bearing | gold absent |
|---|---|---|
| R1 dense | +11 / −1 | +56 / −103 |
| R0 | +6 / −3 | +40 / −104 |
| R3 MedCorp MedCPT | +5 / −1 | +54 / −92 |
| R3 MedCorp hybrid + CE | +7 / −4 | +50 / −118 |
| R3 textbooks+StatPearls MedCPT + CE | +8 / −2 | +58 / −112 |
| R3 MedCorp option-aware + CE | +9 / −8 | +49 / −147 |
| R4 | +4 / −0 | +49 / −98 |
| R2 | +3 / −4 | +46 / −151 |

The option-aware rows (R2, and R2+R3) show that a passage supporting a distractor actively misleads the model.

## Contamination
- **Exact scan of the full corpora** (wiki 33.1M, ARC 14.6M, MedRAG 24.4M), looking for the normalized question prefix (≤150 chars) of 572 questions: 0 hits.
- **8-gram overlap in the retrieved text** (top-5 passages per config):
  - ARC Corpus: 4–6 of 74 SciQ questions have ≥50% overlap, and 1 has ≥90%.
  - Wikipedia: none.
  - R3: one Validation MedQA question (row 206) has 63% overlap with a PubMed snippet in the full-MedCorp configs, probably the case report it was written from. Nothing else in R3 reaches 30%, and nothing on Test does.

## Costs and practicality

**Corpora built:**
- **Wikipedia:** 33,128,679 chunks of ~100 words.
  - BM25: 3.5k CPU-s.
  - bge-base embedding: 5.0 H100-hours (~1.85k passages/s per GPU), ~51 GB fp16.
- **ARC Corpus:** 14.6M sentences; 0.5 H100-hours.
- **MedRAG (MedCorp):** 24,408,616 snippets in 61 parts.
  - Snippets: textbooks 125,847 + StatPearls 384,068 + PubMed 23,898,701.
  - StatPearls was rebuilt from the NCBI tarball with MedRAG's chunker, because HF `MedRAG/statpearls` holds no data for licence reasons.
  - MedCPT embedding: ~900 docs/s per H100 (parts 0–55, 2026-11-05) and ~220 docs/s per L4 (parts 56–59, 2026-11-18, ~40 min per 530k-snippet part).

**Retrieval for ~3,300 queries per corpus** (base + option-aware queries, Val+Test):
- Wikipedia/ARC: BM25 ~175 s over 41 CPU containers, exact dense search on CPU ~150 s, cross-encoder on 150k pairs ~84 s on ≤3 H100s.
- Full MedCorp: BM25 132 s over 61 CPU containers, dense search 268 s, 70,939 passages fetched, MedCPT-CE on 157,103 pairs ~7 min on ≤3 L4s. 847 s in total.
- Textbooks+StatPearls: 154,003 CE pairs, 889 s in total (dominated by reranking on L4s).

**R3 run (2026-11-18).**
- `--stage r3` / `launch_r3.py` on the deployed app ran missing embeddings → retrieval → contexts → inference → NLI in 2.6 h of wall time.
- Inference per split (8 options × 300 rows): 3B ~6 min, 7B ~13 min, 14B ~25 min on L4; 32B ~17 min on L40S.
- The first attempt ran with `modal run --detach`. After ~45 min its orchestrator input was cancelled (cause not determined), and the ephemeral app then shut down 3 in-flight embedding parts. Running the orchestrator as a spawned call on a **deployed** app avoids that.

**Online cost per question:**
- One query encode, plus ~50 cross-encoder pairs if reranked.
- One label-only pass with ~400 extra prompt tokens.
- Option-aware retrieval multiplies the number of queries by 1 + the number of options.
- A MedCorp index is ~37 GB of fp16 MedCPT embeddings plus the BM25 postings.

**GPUs.**
- Early fan-out jobs hit the workspace's 10-GPU cap and queued.
- The fixes:
  - explicit per-function limits (`GPU_BUDGET` in `rag_eval_modal.py`);
  - query encoding and dense search moved to CPU;
  - bigger batches;
  - phases run in order.
- The first R1/R2/R4/R5 runs used H100s, and that was the main cost driver.
- `rag_eval_modal.py` now uses `GPU_SMALL = "L4"` for encoders, rerankers, NLI and Qwen 3B–14B, and `GPU_BIG = "L40S"` for Qwen-32B, whose NF4 weights are too tight for an L4.

**Licensing and availability:**
- Wikipedia: CC BY-SA.
- StatPearls: NC-ND, not redistributable, so it was rebuilt from NCBI.
- MedRAG textbooks and PubMed: on HF.
- ARC Corpus: AI2 S3 zip.

## Viability verdicts

| option | verdict |
|---|---|
| R0 | drop |
| R1 BM25 | drop (dense is better) |
| R1 dense | best retrieval option, but not net-positive; use only behind a gate |
| reranker | not worth it (helps R3 a little, hurts R1) |
| 3000-char context | no better than 1500; worse for R3 |
| R2 option-aware | drop (most harmful, on Wikipedia and on MedCorp) |
| R3 MedCorp (full) | drop: −2.4 to −4.3 always-on, hurts MedQA at every size, ~37 GB index |
| R3 textbooks+StatPearls | drop: within noise of R1 dense; its Validation MedQA gain did not replicate on Test |
| R4 ARC Corpus | drop |
| R5-32B | within noise |
| R5 self-generated | drop |
| R6 | sanity check only |
| gates | remove the harm but give no gain; keep the AQUA-RAT/ReClor routing |

## Remaining (optional)
- **The Train split:** retrieval and inference for the chosen options on the 2,400 Train rows. It is only needed to train a learned injection gate. Given that every fitted gate converges to "inject almost never", that gate is unlikely to change the verdict.

## Files and protocol
- **Code:**
  - `rag_eval/rag_eval_modal.py`. `--stage r3` runs R3 end to end and is resumable; `orchestrate_r3` and `build_contexts_remote` are in this file.
  - `rag_eval/launch_r3.py` spawns that orchestrator on the deployed app (`modal deploy rag_eval/rag_eval_modal.py` first).
  - Also `statpearls_chunk.py`, `pull.sh` and `rag_eval/analysis/*.py`.
- **Analysis (local, no venv needed):** `uv run --with pandas,numpy python rag_eval/analysis/<script>.py`.
  - `stage2_analysis.py` takes the option list as arguments. The list used is in `results/stage2_summary.txt`.
  - `make_tables.py > results/tables.md`.
- **Colab port** (kept, but not needed any more): `colab/R2R_RAG_Colab_<size>.ipynb` and `colab/make_kit.sh`.
- **Results** in `rag_eval/results/`:
  - Stage 1 and 2 tables: `stage1_recall.csv`, `stage2_accuracy.csv`, `stage2_per_source.csv`, `tables.md`
  - Per-row outputs: `stage2_rows/`, `runs/`
  - Retrieval inputs: `retrieval/{wiki,arc,med,medtb}/`, `gen/`, `contexts/`
  - Gate and contamination data: `nli.json`, `contam.json`, `contam_retrieved.csv`, `answer_bearing.csv`
- **Prompts and inference:** the same reflector_core calls (`build_messages_direct` + `infer_label_only`) and the same 4-bit NF4 weights as the main pipeline.
- **Context format:** `[title]\ntext`, 1500 chars, top-3 passages (top-5 for ARC sentences). The `__L3000` variants use 3000 chars and top-5 passages.
- **R3 specifics:**
  - Encoders: MedCPT query/article encoders and MedCPT-Cross-Encoder.
  - BM25 idf for the textbooks+StatPearls subset comes from the full MedCorp, so its scores are not a standalone index.
