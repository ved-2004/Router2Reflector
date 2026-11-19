
#### Test delta accuracy (pp) vs direct, gate = `always`  [paired-bootstrap 95% CI]

| option | 3B | 7B | 14B | 32B | mean | inject % |
|---|---|---|---|---|---|---|
| R6 oracle SciQ support passage | +0.7 [+0.0, +1.7] | +1.0 [+0.0, +2.3] | +1.0 [+0.0, +2.3] | +0.7 [+0.0, +1.7] | +0.8 | 15 |
| R5 generate-then-read (32B writes) | +0.3 [-4.3, +5.0] | +1.0 [-2.3, +4.3] | -1.7 [-4.7, +1.3] | -3.0 [-6.0, -0.3] | -0.8 | 100 |
| R1 wiki dense, 5 psg / 3000 chars | -4.0 [-8.3, +0.0] | -1.0 [-4.3, +2.7] | -1.7 [-4.7, +1.3] | -1.7 [-4.0, +0.7] | -2.1 | 100 |
| R5 self-generated (each size writes) | -4.7 [-9.0, -0.3] | +0.0 [-3.0, +3.3] | -0.7 [-4.3, +3.0] | -3.0 [-6.0, -0.3] | -2.1 | 100 |
| R1 wiki full-article, dense bge | -3.3 [-7.3, +0.7] | -1.3 [-4.3, +1.7] | -2.3 [-5.7, +0.7] | -2.3 [-5.0, +0.3] | -2.3 | 100 |
| R3 MedCorp MedCPT + MedCPT-CE | -4.7 [-8.3, -1.0] | -1.7 [-5.0, +2.0] | -1.7 [-4.7, +1.3] | -1.7 [-4.0, +0.3] | -2.4 | 100 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | -2.7 [-6.3, +0.7] | -1.7 [-4.7, +1.0] | -3.0 [-6.7, +0.0] | -3.0 [-5.7, -0.3] | -2.6 | 100 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | -4.7 [-8.7, -0.7] | -2.0 [-5.3, +1.3] | -1.7 [-4.3, +1.0] | -3.3 [-5.7, -1.3] | -2.9 | 100 |
| R4 ARC corpus dense bge (5 sent.) | -4.3 [-7.7, -1.3] | -2.3 [-5.3, +0.3] | -1.7 [-5.0, +1.3] | -4.0 [-6.7, -1.3] | -3.1 | 100 |
| R3 textbooks+StatPearls MedCPT + CE | -5.0 [-8.7, -1.3] | -2.7 [-5.7, +0.7] | -3.0 [-6.3, +0.3] | -2.3 [-5.0, +0.0] | -3.2 | 100 |
| R3 MedCorp BM25 + MedCPT-CE | -5.0 [-9.3, -0.7] | -1.7 [-5.3, +2.0] | -3.3 [-6.7, -0.3] | -3.3 [-6.0, -1.0] | -3.3 | 100 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | -5.3 [-9.3, -1.3] | -2.3 [-6.0, +1.3] | -3.0 [-6.3, +0.0] | -2.7 [-5.0, -0.7] | -3.3 | 100 |
| R0 BM25 wiki leads (current, filtered) | -3.7 [-7.3, -0.3] | -2.0 [-4.7, +1.0] | -4.0 [-7.3, -0.7] | -3.7 [-6.3, -1.0] | -3.3 | 82 |
| R1 wiki dense + bge-reranker | -3.0 [-7.0, +0.7] | -2.7 [-6.0, +0.7] | -4.7 [-8.3, -1.3] | -4.0 [-7.0, -1.0] | -3.6 | 100 |
| R2+R3 MedCorp option-aware + MedCPT-CE | -6.3 [-10.7, -1.7] | -2.3 [-6.0, +1.3] | -4.0 [-7.7, -0.7] | -4.0 [-7.0, -1.3] | -4.2 | 100 |
| R4 ARC option-aware hybrid + reranker | -6.0 [-10.3, -1.7] | -3.0 [-6.3, +0.3] | -3.7 [-7.0, -0.3] | -4.0 [-6.7, -1.3] | -4.2 | 100 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | -7.3 [-12.0, -3.0] | -2.7 [-6.0, +0.7] | -3.7 [-7.0, -0.3] | -3.7 [-6.0, -1.7] | -4.3 | 100 |
| R2 wiki option-aware hybrid + reranker | -10.3 [-15.0, -5.7] | -6.7 [-10.7, -3.0] | -4.3 [-7.7, -1.0] | -4.3 [-7.3, -1.7] | -6.4 | 100 |

#### Test delta accuracy (pp) vs direct, gate = `route`  [paired-bootstrap 95% CI]

| option | 3B | 7B | 14B | 32B | mean | inject % |
|---|---|---|---|---|---|---|
| R6 oracle SciQ support passage | +0.7 [+0.0, +1.7] | +1.0 [+0.0, +2.3] | +1.0 [+0.0, +2.3] | +0.7 [+0.0, +1.7] | +0.8 | 15 |
| R5 generate-then-read (32B writes) | +0.0 [-3.3, +3.3] | +1.3 [-1.0, +4.0] | +0.0 [-2.0, +2.0] | -1.3 [-3.7, +0.7] | +0.0 | 67 |
| R1 wiki full-article, dense bge | -2.0 [-5.0, +0.7] | -0.7 [-3.0, +1.7] | -1.0 [-3.7, +1.3] | -1.0 [-3.3, +1.0] | -1.2 | 67 |
| R5 self-generated (each size writes) | -4.3 [-7.7, -1.0] | +0.3 [-2.3, +3.0] | +0.0 [-2.0, +2.0] | -1.3 [-3.7, +0.7] | -1.3 | 67 |
| R1 wiki dense, 5 psg / 3000 chars | -3.3 [-6.3, -0.7] | -1.7 [-4.0, +1.0] | +0.0 [-2.7, +2.7] | -0.7 [-2.3, +1.0] | -1.4 | 67 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | -1.7 [-4.7, +1.0] | -1.7 [-3.7, +0.3] | -2.7 [-5.3, +0.3] | -1.3 [-3.7, +1.0] | -1.8 | 67 |
| R3 textbooks+StatPearls MedCPT + CE | -2.7 [-5.3, +0.0] | -2.7 [-5.0, -0.3] | -2.0 [-4.7, +0.7] | -1.3 [-3.3, +0.7] | -2.2 | 67 |
| R3 MedCorp MedCPT + MedCPT-CE | -3.7 [-6.3, -1.0] | -2.3 [-5.0, +0.3] | -1.7 [-4.0, +0.7] | -1.3 [-3.3, +0.3] | -2.2 | 67 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | -3.7 [-6.3, -1.0] | -2.3 [-5.0, +0.3] | -1.0 [-3.7, +1.3] | -2.0 [-4.0, -0.3] | -2.2 | 67 |
| R1 wiki dense + bge-reranker | -1.7 [-4.3, +0.7] | -2.0 [-4.3, +0.3] | -3.0 [-6.0, -0.3] | -2.7 [-5.3, +0.0] | -2.3 | 67 |
| R0 BM25 wiki leads (current, filtered) | -3.7 [-6.3, -1.0] | -0.7 [-3.0, +1.3] | -2.3 [-5.3, +0.7] | -2.7 [-5.0, -0.7] | -2.3 | 57 |
| R4 ARC corpus dense bge (5 sent.) | -3.7 [-6.3, -1.0] | -2.3 [-4.7, +0.0] | -1.0 [-3.3, +1.3] | -3.0 [-5.3, -1.0] | -2.5 | 67 |
| R4 ARC option-aware hybrid + reranker | -5.0 [-8.0, -2.3] | -1.7 [-4.3, +0.7] | -1.7 [-4.3, +1.0] | -1.7 [-4.0, +0.3] | -2.5 | 67 |
| R3 MedCorp BM25 + MedCPT-CE | -5.0 [-8.0, -2.0] | -2.3 [-5.3, +0.3] | -3.7 [-6.7, -1.0] | -2.0 [-4.0, -0.3] | -3.2 | 67 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | -6.0 [-9.7, -3.0] | -3.0 [-6.0, -0.3] | -2.7 [-5.7, +0.0] | -2.3 [-4.3, -0.7] | -3.5 | 67 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | -5.7 [-9.0, -2.7] | -3.0 [-5.7, +0.0] | -3.3 [-6.0, -0.7] | -2.0 [-4.3, +0.0] | -3.5 | 67 |
| R2+R3 MedCorp option-aware + MedCPT-CE | -6.0 [-9.7, -2.7] | -4.0 [-6.7, -1.3] | -4.3 [-7.3, -1.3] | -3.0 [-5.3, -0.7] | -4.3 | 67 |
| R2 wiki option-aware hybrid + reranker | -7.3 [-11.0, -4.0] | -5.0 [-8.0, -2.7] | -3.0 [-6.0, -0.3] | -3.7 [-6.3, -1.0] | -4.8 | 67 |

#### Test delta accuracy (pp) vs direct, gate = `H`  [paired-bootstrap 95% CI]

| option | 3B | 7B | 14B | 32B | mean | inject % |
|---|---|---|---|---|---|---|
| R6 oracle SciQ support passage | +0.7 [+0.0, +1.7] | +0.0 [+0.0, +0.0] | +1.0 [+0.0, +2.3] | +0.0 [+0.0, +0.0] | +0.4 | 4 |
| R5 generate-then-read (32B writes) | -0.7 [-3.7, +2.7] | +1.3 [-1.0, +3.7] | -0.3 [-1.7, +1.0] | +0.0 [-1.7, +1.7] | +0.1 | 16 |
| R5 self-generated (each size writes) | -1.0 [-3.0, +1.0] | +1.0 [-1.3, +3.3] | +0.0 [-1.0, +1.0] | +0.0 [-1.7, +1.7] | +0.0 | 9 |
| R1 wiki full-article, dense bge | -0.7 [-2.7, +1.0] | +0.0 [-1.3, +1.3] | +0.7 [+0.0, +1.7] | +0.0 [+0.0, +0.0] | +0.0 | 6 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | -1.0 [-3.0, +1.0] | +0.7 [+0.0, +1.7] | -0.3 [-2.0, +1.0] | +0.0 [+0.0, +0.0] | -0.2 | 5 |
| R2 wiki option-aware hybrid + reranker | +0.0 [+0.0, +0.0] | -1.0 [-2.3, +0.0] | -0.3 [-1.7, +0.7] | +0.0 [+0.0, +0.0] | -0.3 | 2 |
| R1 wiki dense, 5 psg / 3000 chars | -2.0 [-4.0, +0.0] | +0.0 [+0.0, +0.0] | +0.7 [+0.0, +1.7] | +0.0 [+0.0, +0.0] | -0.3 | 4 |
| R2+R3 MedCorp option-aware + MedCPT-CE | -1.0 [-3.3, +1.3] | -0.3 [-1.7, +1.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.3 | 5 |
| R1 wiki dense + bge-reranker | -1.0 [-3.0, +1.0] | -0.7 [-2.7, +1.3] | +0.0 [-1.3, +1.3] | +0.0 [+0.0, +0.0] | -0.4 | 10 |
| R3 textbooks+StatPearls MedCPT + CE | -0.7 [-2.7, +1.0] | -1.0 [-3.0, +1.0] | +0.0 [-1.3, +1.3] | -0.3 [-2.0, +1.3] | -0.5 | 10 |
| R3 MedCorp MedCPT + MedCPT-CE | -1.0 [-2.7, +0.7] | +0.0 [+0.0, +0.0] | +0.3 [-0.7, +1.7] | -1.3 [-3.3, +0.7] | -0.5 | 7 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | -2.3 [-4.7, -0.3] | +0.0 [+0.0, +0.0] | +0.3 [-0.7, +1.7] | +0.0 [+0.0, +0.0] | -0.5 | 4 |
| R3 MedCorp BM25 + MedCPT-CE | -2.3 [-4.7, +0.0] | +0.0 [+0.0, +0.0] | +0.3 [-0.7, +1.3] | +0.0 [+0.0, +0.0] | -0.5 | 4 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | -1.0 [-3.3, +1.3] | -0.7 [-2.0, +0.7] | +0.0 [-1.3, +1.3] | -0.7 [-2.0, +0.7] | -0.6 | 5 |
| R0 BM25 wiki leads (current, filtered) | -1.7 [-3.3, +0.0] | +0.0 [+0.0, +0.0] | +0.3 [-0.7, +1.3] | -1.0 [-2.7, +0.7] | -0.6 | 5 |
| R4 ARC corpus dense bge (5 sent.) | -2.7 [-5.3, -0.3] | -1.0 [-3.0, +0.7] | +0.3 [-0.7, +1.3] | +0.0 [+0.0, +0.0] | -0.8 | 11 |
| R4 ARC option-aware hybrid + reranker | -3.0 [-5.3, -0.7] | -0.7 [-1.7, +0.0] | +0.3 [-0.7, +1.3] | +0.0 [+0.0, +0.0] | -0.8 | 5 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | -1.7 [-3.7, +0.0] | +0.0 [+0.0, +0.0] | -0.3 [-2.0, +1.3] | -2.0 [-4.0, -0.3] | -1.0 | 7 |

#### Test delta accuracy (pp) vs direct, gate = `nli`  [paired-bootstrap 95% CI]

| option | 3B | 7B | 14B | 32B | mean | inject % |
|---|---|---|---|---|---|---|
| R5 generate-then-read (32B writes) | +0.0 [-3.3, +3.3] | +1.0 [-1.0, +3.0] | +0.3 [+0.0, +1.0] | -1.3 [-3.3, +0.7] | +0.0 | 56 |
| R1 wiki full-article, dense bge | +0.0 [+0.0, +0.0] | -0.3 [-1.7, +0.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.1 | 8 |
| R4 ARC corpus dense bge (5 sent.) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.3 [-1.0, +0.0] | -0.3 [-1.0, +0.0] | -0.2 | 7 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | +0.0 [+0.0, +0.0] | -0.7 [-1.7, +0.0] | +0.3 [+0.0, +1.0] | -0.3 [-1.0, +0.0] | -0.2 | 8 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | -0.3 [-1.0, +0.0] | +0.0 [+0.0, +0.0] | -0.3 [-1.3, +0.7] | -0.3 [-1.0, +0.0] | -0.2 | 6 |
| R3 MedCorp BM25 + MedCPT-CE | -0.3 [-1.0, +0.0] | -0.7 [-1.7, +0.0] | +0.0 [+0.0, +0.0] | -0.3 [-1.0, +0.0] | -0.3 | 7 |
| R3 textbooks+StatPearls MedCPT + CE | -0.3 [-1.0, +0.0] | +0.3 [+0.0, +1.0] | -1.0 [-3.0, +1.0] | -0.3 [-2.0, +1.0] | -0.3 | 15 |
| R3 MedCorp MedCPT + MedCPT-CE | -0.3 [-1.7, +0.7] | -1.0 [-2.3, +0.3] | -0.3 [-1.3, +0.7] | -0.3 [-1.0, +0.0] | -0.5 | 23 |
| R4 ARC option-aware hybrid + reranker | +0.0 [+0.0, +0.0] | -1.3 [-3.3, +0.3] | -0.7 [-1.7, +0.0] | +0.0 [+0.0, +0.0] | -0.5 | 9 |
| R2 wiki option-aware hybrid + reranker | -1.0 [-2.3, +0.0] | -1.0 [-2.3, +0.0] | -0.3 [-1.0, +0.0] | +0.0 [+0.0, +0.0] | -0.6 | 8 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | -0.3 [-1.3, +0.7] | -0.3 [-2.0, +1.3] | +0.0 [+0.0, +0.0] | -2.0 [-4.0, -0.3] | -0.7 | 28 |
| R2+R3 MedCorp option-aware + MedCPT-CE | -1.7 [-3.3, -0.3] | -0.7 [-1.7, +0.0] | -0.3 [-1.0, +0.0] | +0.0 [+0.0, +0.0] | -0.7 | 6 |
| R5 self-generated (each size writes) | -1.3 [-3.0, +0.0] | -0.7 [-3.0, +1.3] | +0.3 [+0.0, +1.0] | -1.3 [-3.3, +0.7] | -0.8 | 36 |
| R1 wiki dense, 5 psg / 3000 chars | -3.0 [-6.0, -0.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.8 | 17 |
| R1 wiki dense + bge-reranker | -1.0 [-3.0, +1.0] | -1.0 [-2.3, +0.0] | -1.0 [-2.7, +0.7] | -0.3 [-1.0, +0.0] | -0.8 | 28 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | -6.0 [-9.7, -2.7] | -0.3 [-1.0, +0.0] | +0.3 [+0.0, +1.0] | -0.3 [-1.0, +0.0] | -1.6 | 25 |

#### Test delta accuracy (pp) vs direct, gate = `src`  [paired-bootstrap 95% CI]

| option | 3B | 7B | 14B | 32B | mean | inject % |
|---|---|---|---|---|---|---|
| R6 oracle SciQ support passage | +0.7 [+0.0, +1.7] | +1.0 [+0.0, +2.3] | +1.0 [+0.0, +2.3] | +0.0 [+0.0, +0.0] | +0.7 | 11 |
| R2 wiki option-aware hybrid + reranker | +0.0 [+0.0, +0.0] | -0.7 [-2.0, +0.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.2 | 4 |
| R2+R3 MedCorp option-aware + MedCPT-CE | -1.3 [-4.0, +1.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.3 | 4 |
| R5 generate-then-read (32B writes) | +0.0 [-3.3, +3.3] | +1.3 [-0.7, +3.3] | -0.3 [-2.7, +1.7] | -2.3 [-4.3, -0.3] | -0.3 | 50 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | +0.0 [-1.7, +2.0] | +0.0 [-1.3, +1.3] | -1.3 [-3.7, +0.7] | +0.0 [+0.0, +0.0] | -0.3 | 17 |
| R1 wiki dense, 5 psg / 3000 chars | -2.0 [-4.3, +0.0] | +0.0 [-1.0, +1.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.5 | 12 |
| R3 MedCorp MedCPT + MedCPT-CE | -0.7 [-2.7, +1.0] | +0.0 [+0.0, +0.0] | +0.0 [-1.7, +1.7] | -1.3 [-3.0, +0.0] | -0.5 | 17 |
| R0 BM25 wiki leads (current, filtered) | -1.0 [-2.7, +0.7] | +0.0 [+0.0, +0.0] | -1.0 [-2.3, +0.0] | -0.3 [-1.0, +0.0] | -0.6 | 10 |
| R4 ARC option-aware hybrid + reranker | -0.7 [-2.0, +0.7] | +0.0 [+0.0, +0.0] | -1.7 [-3.7, +0.3] | +0.0 [+0.0, +0.0] | -0.6 | 8 |
| R5 self-generated (each size writes) | -2.3 [-6.0, +1.3] | +1.3 [-1.3, +4.0] | +0.7 [-1.7, +3.0] | -2.3 [-4.3, -0.3] | -0.7 | 46 |
| R4 ARC corpus dense bge (5 sent.) | -2.3 [-4.7, -0.3] | -0.7 [-2.0, +0.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.8 | 17 |
| R1 wiki full-article, dense bge | +0.0 [-1.7, +1.7] | +0.3 [-1.3, +2.0] | -1.7 [-3.7, +0.3] | -1.7 [-3.7, +0.0] | -0.8 | 25 |
| R3 MedCorp BM25 + MedCPT-CE | -1.3 [-3.3, +0.3] | +0.0 [+0.0, +0.0] | -2.3 [-4.7, -0.3] | +0.0 [+0.0, +0.0] | -0.9 | 12 |
| R1 wiki dense + bge-reranker | -0.7 [-2.3, +1.0] | +0.0 [+0.0, +0.0] | -3.0 [-5.7, -0.3] | +0.0 [+0.0, +0.0] | -0.9 | 17 |
| R3 textbooks+StatPearls MedCPT + CE | -1.0 [-3.0, +1.0] | -1.0 [-3.0, +0.7] | -1.7 [-4.3, +1.0] | -1.0 [-2.7, +0.7] | -1.2 | 25 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | -3.0 [-6.0, +0.0] | +0.0 [+0.0, +0.0] | -0.3 [-1.3, +0.7] | -1.7 [-3.3, -0.3] | -1.2 | 29 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | -2.0 [-4.0, -0.3] | +0.0 [+0.0, +0.0] | -2.0 [-4.0, +0.0] | -1.3 [-3.0, +0.3] | -1.3 | 13 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | -3.7 [-6.7, -1.0] | -0.3 [-1.0, +0.0] | -0.7 [-1.7, +0.0] | -2.0 [-3.7, -0.7] | -1.7 | 21 |

#### Fixed / broken counts (Test, always-on)

| option | 3B | 7B | 14B | 32B |
|---|---|---|---|---|
| R0 BM25 wiki leads (current, filtered) | +9/-20 | +7/-13 | +6/-18 | +3/-14 |
| R4 ARC corpus dense bge (5 sent.) | +7/-20 | +6/-13 | +10/-15 | +2/-14 |
| R4 ARC option-aware hybrid + reranker | +13/-31 | +9/-18 | +8/-19 | +3/-15 |
| R5 generate-then-read (32B writes) | +26/-25 | +14/-11 | +8/-13 | +5/-14 |
| R5 self-generated (each size writes) | +16/-30 | +12/-12 | +12/-14 | +5/-14 |
| R3 MedCorp BM25 + MedCPT-CE | +15/-30 | +12/-17 | +7/-17 | +2/-12 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | +12/-26 | +10/-16 | +6/-11 | +1/-11 |
| R3 MedCorp MedCPT + MedCPT-CE | +10/-24 | +12/-17 | +8/-13 | +3/-8 |
| R2+R3 MedCorp option-aware + MedCPT-CE | +16/-35 | +12/-19 | +8/-20 | +3/-15 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | +12/-28 | +12/-19 | +8/-17 | +2/-10 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | +12/-34 | +11/-19 | +8/-19 | +1/-12 |
| R3 textbooks+StatPearls MedCPT + CE | +10/-25 | +7/-15 | +9/-18 | +4/-11 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | +11/-19 | +6/-11 | +9/-18 | +4/-13 |
| R6 oracle SciQ support passage | +2/-0 | +3/-0 | +3/-0 | +2/-0 |
| R1 wiki full-article, dense bge | +13/-23 | +9/-13 | +9/-16 | +6/-13 |
| R1 wiki dense, 5 psg / 3000 chars | +15/-27 | +12/-15 | +8/-13 | +4/-9 |
| R1 wiki dense + bge-reranker | +13/-22 | +8/-16 | +8/-22 | +4/-16 |
| R2 wiki option-aware hybrid + reranker | +12/-43 | +7/-27 | +8/-21 | +3/-16 |

#### Per-source Test delta (pp, always-on; n=50 per source, 1 question = 2 pp)

| option | size | AQUA-RAT | ARC-Easy | MMLU | MedQA | ReClor | SciQ |
|---|---|---|---|---|---|---|---|
| R0 BM25 wiki leads (current, filtered) | 3B | -2 | -8 | -4 | -6 | +2 | -4 |
| R0 BM25 wiki leads (current, filtered) | 7B | +0 | +0 | +0 | -4 | -8 | +0 |
| R0 BM25 wiki leads (current, filtered) | 14B | -6 | -8 | -2 | -4 | -4 | +0 |
| R0 BM25 wiki leads (current, filtered) | 32B | -2 | -2 | -4 | -10 | -4 | +0 |
| R1 wiki full-article, dense bge | 3B | -6 | -6 | +0 | +0 | -2 | -6 |
| R1 wiki full-article, dense bge | 7B | -2 | -2 | -2 | -4 | -2 | +4 |
| R1 wiki full-article, dense bge | 14B | -2 | -2 | -4 | -4 | -6 | +4 |
| R1 wiki full-article, dense bge | 32B | -6 | +0 | +0 | -10 | -2 | +4 |
| R1 wiki dense + bge-reranker | 3B | -2 | -2 | +2 | -2 | -6 | -8 |
| R1 wiki dense + bge-reranker | 7B | -2 | -2 | +0 | -10 | -2 | +0 |
| R1 wiki dense + bge-reranker | 14B | -4 | -4 | -4 | -12 | -6 | +2 |
| R1 wiki dense + bge-reranker | 32B | -6 | +0 | -4 | -12 | -2 | +0 |
| R1 wiki dense, 5 psg / 3000 chars | 3B | -2 | -6 | -4 | -6 | -2 | -4 |
| R1 wiki dense, 5 psg / 3000 chars | 7B | +6 | -4 | +0 | -10 | -2 | +4 |
| R1 wiki dense, 5 psg / 3000 chars | 14B | -6 | -2 | +0 | +0 | -4 | +2 |
| R1 wiki dense, 5 psg / 3000 chars | 32B | -6 | -2 | +0 | -6 | +0 | +4 |
| R2 wiki option-aware hybrid + reranker | 3B | -4 | -12 | +0 | -26 | -14 | -6 |
| R2 wiki option-aware hybrid + reranker | 7B | -2 | -10 | +0 | -16 | -8 | -4 |
| R2 wiki option-aware hybrid + reranker | 14B | -4 | -4 | -4 | -12 | -4 | +2 |
| R2 wiki option-aware hybrid + reranker | 32B | -2 | -4 | -2 | -16 | -2 | +0 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | 3B | -2 | -8 | +0 | -8 | -4 | -6 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | 7B | +6 | -6 | +0 | -4 | -4 | -4 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | 14B | -2 | -2 | -4 | +0 | -2 | +0 |
| R3 MedCorp (tb+SP+PubMed) MedCPT | 32B | -4 | +0 | -4 | -10 | -4 | +2 |
| R3 MedCorp MedCPT + MedCPT-CE | 3B | +4 | -8 | -2 | -4 | -10 | -8 |
| R3 MedCorp MedCPT + MedCPT-CE | 7B | +10 | -8 | -4 | -6 | -6 | +4 |
| R3 MedCorp MedCPT + MedCPT-CE | 14B | +4 | -4 | +0 | -6 | -4 | +0 |
| R3 MedCorp MedCPT + MedCPT-CE | 32B | -4 | +0 | -2 | -8 | +2 | +2 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | 3B | +6 | -8 | -4 | -12 | -4 | -10 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | 7B | +6 | -6 | -6 | +0 | -2 | -6 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | 14B | +2 | -6 | -2 | -12 | +0 | +0 |
| R3 MedCorp BM25+MedCPT + MedCPT-CE | 32B | -4 | -2 | +0 | -8 | +0 | -2 |
| R2+R3 MedCorp option-aware + MedCPT-CE | 3B | +0 | -10 | -6 | -8 | -2 | -12 |
| R2+R3 MedCorp option-aware + MedCPT-CE | 7B | +10 | -2 | -8 | -12 | +0 | -2 |
| R2+R3 MedCorp option-aware + MedCPT-CE | 14B | +6 | -4 | -6 | -14 | -4 | -2 |
| R2+R3 MedCorp option-aware + MedCPT-CE | 32B | -6 | -2 | -4 | -14 | +0 | +2 |
| R3 MedCorp BM25 + MedCPT-CE | 3B | +8 | -8 | -4 | -8 | -8 | -10 |
| R3 MedCorp BM25 + MedCPT-CE | 7B | +6 | -4 | -4 | +2 | -2 | -8 |
| R3 MedCorp BM25 + MedCPT-CE | 14B | +2 | -6 | -2 | -14 | +0 | +0 |
| R3 MedCorp BM25 + MedCPT-CE | 32B | -6 | -2 | +0 | -8 | -2 | -2 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | 3B | -6 | -10 | +0 | -12 | -2 | -14 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | 7B | +6 | -6 | -2 | -6 | -4 | -4 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | 14B | -2 | -8 | -2 | -4 | -4 | -2 |
| R3 MedCorp hybrid + CE, 5 psg / 3000 chars | 32B | -8 | -2 | +0 | -12 | +0 | +0 |
| R3 textbooks+StatPearls MedCPT + CE | 3B | -10 | -4 | +0 | -6 | -4 | -6 |
| R3 textbooks+StatPearls MedCPT + CE | 7B | +8 | -4 | -2 | -6 | -8 | -4 |
| R3 textbooks+StatPearls MedCPT + CE | 14B | +2 | -4 | -2 | -4 | -8 | -2 |
| R3 textbooks+StatPearls MedCPT + CE | 32B | -8 | +0 | -2 | -6 | +2 | +0 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | 3B | -4 | -6 | +2 | +0 | -2 | -6 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | 7B | +8 | -4 | -2 | +0 | -8 | -4 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | 14B | +4 | -8 | -2 | -2 | -6 | -4 |
| R3 textbooks+StatPearls BM25+MedCPT + CE | 32B | -8 | -2 | -2 | -4 | -2 | +0 |
| R4 ARC corpus dense bge (5 sent.) | 3B | +0 | -4 | -4 | -10 | -4 | -4 |
| R4 ARC corpus dense bge (5 sent.) | 7B | +0 | -4 | -2 | -8 | +0 | +0 |
| R4 ARC corpus dense bge (5 sent.) | 14B | +0 | -2 | -2 | +0 | -4 | -2 |
| R4 ARC corpus dense bge (5 sent.) | 32B | -6 | -2 | -2 | -12 | +0 | -2 |
| R4 ARC option-aware hybrid + reranker | 3B | +2 | -4 | -2 | -16 | -8 | -8 |
| R4 ARC option-aware hybrid + reranker | 7B | -4 | +0 | -4 | -8 | -4 | +2 |
| R4 ARC option-aware hybrid + reranker | 14B | -4 | +0 | -4 | -10 | -8 | +4 |
| R4 ARC option-aware hybrid + reranker | 32B | -6 | +0 | -2 | -12 | -8 | +4 |
| R5 generate-then-read (32B writes) | 3B | +2 | -4 | +6 | +2 | +0 | -4 |
| R5 generate-then-read (32B writes) | 7B | +0 | +2 | +0 | +6 | -2 | +0 |
| R5 generate-then-read (32B writes) | 14B | -2 | +0 | -2 | +0 | -8 | +2 |
| R5 generate-then-read (32B writes) | 32B | -6 | +0 | +0 | -8 | -4 | +0 |
| R6 oracle SciQ support passage | 3B | +0 | +0 | +0 | +0 | +0 | +4 |
| R6 oracle SciQ support passage | 7B | +0 | +0 | +0 | +0 | +0 | +6 |
| R6 oracle SciQ support passage | 14B | +0 | +0 | +0 | +0 | +0 | +6 |
| R6 oracle SciQ support passage | 32B | +0 | +0 | +0 | +0 | +0 | +4 |
