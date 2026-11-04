# Evaluation report

Run: 2026-11-04 | embedder: `st:BAAI/bge-small-en-v1.5` | reranker: `cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2` | generator: `extractive (no LLM)`
Hardware: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel, 8 CPU(s), Python 3.14.6

Corpus: 11 documents / 34 chunks. Questions: 40 (20 dev, 20 test).

## Retrieval (dev)

| configuration | R@1 | R@3 | R@5 | P@3 | P@5 | MRR | NDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dense only | 0.899 | 0.935 | 0.952 | 0.381 | 0.243 | 1.0 | 0.962 | 20.27 | 22.54 |
| BM25 only | 0.732 | 0.917 | 0.982 | 0.357 | 0.257 | 0.881 | 0.906 | 0.07 | 0.16 |
| hybrid (RRF) | 0.875 | 0.935 | 1.0 | 0.381 | 0.271 | 0.964 | 0.977 | 17.58 | 21.8 |
| hybrid + rerank | 0.827 | 0.982 | 1.0 | 0.429 | 0.271 | 0.964 | 0.974 | 382.27 | 440.23 |

## Retrieval (test)

| configuration | R@1 | R@3 | R@5 | P@3 | P@5 | MRR | NDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dense only | 0.821 | 1.0 | 1.0 | 0.405 | 0.243 | 0.964 | 0.974 | 23.24 | 29.11 |
| BM25 only | 0.679 | 1.0 | 1.0 | 0.405 | 0.243 | 0.893 | 0.921 | 0.09 | 0.14 |
| hybrid (RRF) | 0.821 | 1.0 | 1.0 | 0.405 | 0.243 | 0.964 | 0.974 | 23.94 | 24.85 |
| hybrid + rerank | 0.893 | 1.0 | 1.0 | 0.405 | 0.243 | 1.0 | 0.994 | 428.65 | 544.79 |

## Answers (dev)

- turns: 22
- answer_correctness: 0.955
- answerable_correct: 0.944
- abstention_correct (should-refuse): 1.0
- false_abstentions (answerable refused): 1
- citation_hit_rate: 1.0
- citation_precision: 0.962
- both_versions_cited (comparisons): 1.0
- faithfulness_proxy (lexical): 1.0
- permission_leaks (must be 0): 0
- answer_latency_ms_p50: 336.6
- answer_latency_ms_p95: 419.01
- by category: comparison 1/1, cross_document 1/1, exact_id 2/2, followup 4/4, format_pdf 1/1, format_xlsx 1/1, lookup 3/3, paraphrase 1/2, permission_allowed 1/1, permission_denied 2/2, unsupported 2/2, version 2/2

## Answers (test)

- turns: 21
- answer_correctness: 0.952
- answerable_correct: 0.938
- abstention_correct (should-refuse): 1.0
- false_abstentions (answerable refused): 0
- citation_hit_rate: 1.0
- citation_precision: 0.929
- both_versions_cited (comparisons): 1.0
- faithfulness_proxy (lexical): 1.0
- permission_leaks (must be 0): 0
- answer_latency_ms_p50: 346.54
- answer_latency_ms_p95: 368.91
- by category: comparison 2/2, cross_document 1/1, exact_id 1/1, followup 1/2, format_docx 1/1, format_html 1/1, lookup 3/3, paraphrase 2/2, permission_allowed 1/1, permission_denied 2/2, unsupported 3/3, version 2/2

## Caveats

- Small synthetic corpus (about 30 chunks): retrieval metrics saturate and differences between configurations are not statistically meaningful. Do not read them as production quality.
- With the hashing embedder, 'dense' retrieval is lexical; it does not capture paraphrase the way BGE does. Re-run with ERAG_EMBEDDER=bge-small (or bge-m3) and ERAG_RERANKER=cross-encoder.
- The lexical reranker is a heuristic stand-in for a cross-encoder.
- Without an LLM the answers are extractive quotes, so faithfulness_proxy is near-trivial; it is a lexical support check, not an entailment judge. Inspect answers by hand and, if you add an LLM judge, treat it as supplementary evidence.
- Latency is measured on this machine only and excludes model download/warm-up.
- The abstention threshold is chosen on the dev split only (--calibrate). Be aware that a few heuristic fixes in rag.py (question-filler words, section context in support scoring) were written while probing questions that also appear in the dev split, so dev scores are optimistic; the test split was not used to tune anything, and its failures (see by-category counts) are left unfixed on purpose.
- Known failure types with the offline stack: vocabulary gaps ('file a claim' vs 'submitted'; 'annual limit' vs 'up to INR 30,000 per calendar year'). A neural embedder, a cross-encoder and an LLM are expected to help, but that has not been measured here.
