# ADR-0003: Reranker Choice

## Status
Accepted

## Context
Citation-required compliance answers need high precision in top contexts.

## Decision
Use cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with lexical fallback.

## Rationale
- Cross-encoders improve top-k relevance ordering versus retrieval-only ranking.
- Lexical fallback preserves service continuity when model download/runtime fails.

## Consequences
- Additional latency step in pipeline.
- Higher faithfulness and lower hallucination risk in compliance use cases.
