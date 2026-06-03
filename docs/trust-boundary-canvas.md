# Trust Boundary Canvas

## Zones
- Zone A (System of Record): Sanitized policy corpus, audit findings, regulatory bulletins.
- Zone B (RAG Compute): Chunking, indexing, retrieval, reranking.
- Zone C (LLM Inference): Prompt construction and answer synthesis.
- Zone D (Human Oversight): Compliance officer review and final approval.

## Capability Placement
- Ingestion and chunking: Zone B
- Retriever and reranker: Zone B
- LLM response generation: Zone C
- Citation verification and confidence scoring: Zone B
- Final recommendation acceptance: Zone D

## Control Points
- Boundary 1 (A -> B): Input sanitation, schema validation, source metadata retention.
- Boundary 2 (B -> C): Redaction checks, strict prompt for citation-only grounding.
- Boundary 3 (C -> D): Mandatory AI recommendation label, confidence, escalation reasons.

## Human-Approve Rules
- Any low-confidence result.
- Any jurisdiction-specific query.
- Any policy-boundary or legal-interpretation query.
- Any answer missing citations.
