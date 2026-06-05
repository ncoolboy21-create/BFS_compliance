# BFS Compliance RAG - PPT Content Pack

## Slide 1 - Title
- BFS Compliance RAG Architecture
- Citation-first compliance assistant with trust controls
- Version: 1.0
- Date: June 2026

## Slide 2 - Problem Statement
- Compliance teams need fast answers across policy, regulation, and audit artifacts.
- Manual lookup is slow and inconsistent under time pressure.
- Pure LLM responses are high-risk without grounded citations.
- Goal: deliver fast, evidence-backed recommendations with explicit human oversight.

## Slide 3 - Target Outcomes
- Reduce time to first draft compliance response.
- Ensure every answer is grounded in retrievable source evidence.
- Provide measurable confidence and escalation reasons.
- Support auditable decision reconstruction.

## Slide 4 - Architecture Snapshot
- FastAPI application layer for API + UI hosting.
- Ingestion pipeline for PDF/TXT/JSON/JSONL documents.
- Hierarchical chunking with overlap.
- ChromaDB persistent vector index with sentence-transformer embeddings.
- Cross-encoder reranker for relevance prioritization.
- LLM generator (OpenAI/Azure/Gemini) with deterministic fallback.
- Trust and escalation service for decision gating.

## Slide 5 - End-to-End Runtime Flow
1. User asks question via UI or `/ask`.
2. Retriever fetches top semantic candidates from ChromaDB.
3. Reranker re-orders candidates by query relevance.
4. Generator produces structured answer + citations.
5. Confidence scorer computes confidence from ranked evidence.
6. Escalation service assigns reasons and trust decision.
7. API returns answer, citations, confidence, trust metadata.

## Slide 6 - Data Ingestion and Lifecycle
- Upload endpoint normalizes incoming files to `SourceDocument`.
- Chunker creates section-aware chunks with configured overlap.
- Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.
- Chunks and metadata are persisted into Chroma collection `compliance_chunks`.
- Uploaded content is queryable immediately and persists across restarts.

## Slide 7 - Key Decision Log (Architecture)

| Decision | Chosen Option | Alternatives Considered | Rationale |
|---|---|---|---|
| API framework | FastAPI | Flask, Django REST | Strong typing, OpenAPI docs, async-ready, rapid iteration |
| Retrieval store | ChromaDB persistent client | FAISS in-memory, Elasticsearch | Low ops burden, native metadata filtering, local persistence |
| Embedding model | all-MiniLM-L6-v2 | larger SBERT models | Good latency/quality tradeoff for on-prem dev machine |
| Reranker | Cross-encoder MiniLM | bi-encoder only, no reranker | Improves precision for top evidence and confidence stability |
| Generator design | Citation-aware JSON output | free-form prose | Structured answer/citation parsing and downstream validation |
| Trust policy | Confidence + escalation reasons | confidence-only gate | Better explainability and safer human-in-the-loop routing |
| Fallback strategy | Deterministic local formatter | hard fail on LLM error | Ensures continuity during provider quota/outage events |
| Upload handling | Normalize to domain model before indexing | raw text passthrough | Preserves metadata and auditability |

## Slide 8 - Key Decision Log (Risk/Trust)

| Decision | Why It Was Needed | Control Implemented | Residual Risk |
|---|---|---|---|
| Human approval threshold at 0.60 | Avoid low-confidence auto-recommendations | trust decision gate in API response | threshold may need calibration per domain |
| Citation-required output | Prevent unsupported claims | citation list in response schema | weak source quality can still mislead |
| Escalation reason taxonomy | Improve explainability | low-confidence/jurisdiction/policy-boundary/missing-citations | false positives in boundary keywords |
| Fallback logging | Make provider failures visible | warning logs with provider error details | logs require active monitoring |

## Slide 9 - Trust Control Logic
- Confidence is derived from reranked evidence scores.
- Decision rule:
  - confidence >= 0.60 -> `AI_AUTO_APPROVED`
  - confidence < 0.60 -> `HUMAN_APPROVAL_REQUIRED`
- Escalation reasons are attached regardless of decision.
- Frontend displays confidence, trust decision, citations, and reasons.

## Slide 10 - Why Outputs Can Still Degrade
- LLM provider quota/rate-limit causes fallback path activation.
- Fallback answers summarize retrieved chunks, not full document pages.
- Retrieval quality depends on chunking and metadata quality.
- Mitigations in place:
  - chunk order reconstruction
  - overlap-aware merging
  - de-duplication of similar details
  - fallback reason logging

## Slide 11 - Operational Readiness
- Health endpoint: `/health`.
- Document listing endpoint: `/documents`.
- Upload endpoint: `/upload`.
- Persistent vector store path: `data/chroma_db`.
- Evaluation assets: `docs/ragas-scorecard.json` and `docs/ragas-scorecard.md`.

## Slide 12 - Metrics for Governance
- Retrieval hit quality (top-k source relevance).
- Citation coverage ratio.
- Low-confidence rate.
- Fallback activation rate by provider.
- Human override/approval rate.
- Mean time to reviewed answer.

## Slide 13 - Deployment Notes
- Recommended start command in dev:
  - `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --reload-exclude .venv`
- Reason for `--reload-exclude .venv`:
  - avoids infinite reload loops when packages are installed/updated.

## Slide 14 - Demo Script (Presenter Notes)
1. Upload compliance PDF.
2. Show `/documents` list update.
3. Ask question and show citations + confidence.
4. Trigger low-confidence case and show human-approval state.
5. Show fallback warning behavior when provider quota fails.

## Slide 15 - Roadmap
- Add source-page preview panel in frontend.
- Add real-time fallback banner in UI.
- Add RPN-based alerting from FMEA register.
- Add provider failover policy and budget guardrails.
- Calibrate confidence threshold on golden-set trends.
