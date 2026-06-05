# Failure Mode Register (FMEA)

Scoring scale:
- Severity (S): 1 low impact -> 10 critical impact
- Occurrence (O): 1 rare -> 10 frequent
- Detection (D): 1 easy to detect -> 10 hard to detect
- RPN = S x O x D

| ID | Process Step | Failure Mode | Potential Effects | Potential Causes | Current Controls | S | O | D | RPN | Recommended Action | Owner | Target Date |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| FMEA-01 | Retrieval -> Generation | Hallucinated or unsupported statement | Incorrect compliance advice; regulatory breach risk | Weak context, model drift, prompt leakage | Citation-required schema, reranker, human gate | 10 | 4 | 6 | 240 | Add strict citation-to-claim validator and reject ungrounded sentences | ML Eng + Compliance | 2026-07-15 |
| FMEA-02 | LLM Inference | Provider quota/rate-limit failure (429) | Fallback mode degrades answer quality | Billing quota exhausted, shared key limits | Warning logs, deterministic fallback | 8 | 7 | 3 | 168 | Add provider budget alerts + UI fallback banner + secondary provider failover | Platform Eng | 2026-06-30 |
| FMEA-03 | Ingestion | Malformed PDF extraction (OCR/noise) | Broken phrases and incomplete answer context | Image-only pages, poor PDF encoding | pypdf extraction + basic cleaning | 7 | 6 | 6 | 252 | Add OCR pipeline and text quality scoring before indexing | Data Eng | 2026-08-01 |
| FMEA-04 | Chunking | Chunk boundary splits key sentence | Missing or fragmented response details | Char-based chunking, overlap boundary artifacts | Overlap + boundary heuristics | 8 | 5 | 5 | 200 | Move to sentence-aware chunking with page semantic markers | ML Eng | 2026-07-20 |
| FMEA-05 | Retrieval | Wrong jurisdiction evidence selected | Non-compliant regional guidance | Missing/incorrect jurisdiction metadata | Jurisdiction filter in retriever | 9 | 3 | 5 | 135 | Add hard jurisdiction mismatch block + metadata completeness checks | Compliance Ops | 2026-07-10 |
| FMEA-06 | Reranking | Relevant chunk ranked too low | Key obligations omitted in final answer | Model mismatch, weak negative sampling | Cross-encoder reranker + LoRA option | 8 | 5 | 6 | 240 | Retrain reranker with hard negatives and regression tests | MLOps | 2026-07-25 |
| FMEA-07 | Trust Decision | Low-quality output auto-approved | Unsafe recommendation accepted | Confidence calibration drift | Confidence threshold + escalation reasons | 9 | 4 | 5 | 180 | Add threshold calibration job and confidence drift monitor | Risk Analytics | 2026-07-05 |
| FMEA-08 | Persistence | Chroma index corruption or partial writes | Missing evidence at query time | Unclean shutdown, filesystem issues | Persistent client + startup load path | 7 | 3 | 7 | 147 | Add collection integrity check + backup/restore routine | Platform Eng | 2026-07-18 |
| FMEA-09 | Auditability | Incomplete decision trace | Cannot reconstruct answer lineage for audit | Missing structured logs for fallback/approval | Basic runtime logs | 8 | 4 | 7 | 224 | Add immutable trace record (query, chunks, scores, provider, decision) | Governance Eng | 2026-07-12 |

## Priority Ranking (Top RPN)
1. FMEA-03 OCR/noisy extraction (RPN 252)
2. FMEA-01 Hallucinated statement (RPN 240)
3. FMEA-06 Reranker miss (RPN 240)
4. FMEA-09 Incomplete audit trace (RPN 224)
5. FMEA-04 Chunk boundary split (RPN 200)

## Immediate Actions (Next 2 Weeks)
1. Implement fallback banner and provider quota alerts (FMEA-02).
2. Add query-level immutable trace logging (FMEA-09).
3. Add sentence-aware chunking prototype and A/B test (FMEA-04).
4. Build citation-to-claim validator guardrail (FMEA-01).
