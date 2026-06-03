# ADR-0002: Embedding and Vector Store

## Status
Accepted

## Context
The initial deliverable targets a synthetic 20-document subset and must remain easy to run locally.

## Decision
- Retrieval baseline: TF-IDF vectorizer + cosine similarity (in-memory)
- Upgrade path: swap with dense embedding index (FAISS / Qdrant) behind retriever interface

## Rationale
- Zero infrastructure dependency for first pass.
- Fast iteration for golden-set tuning.
- Interface remains stable for production replacement.

## Consequences
- Lower semantic recall than dense embeddings.
- Appropriate for this curriculum checkpoint and transparent debugging.
