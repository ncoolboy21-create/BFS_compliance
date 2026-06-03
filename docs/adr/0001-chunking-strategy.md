# ADR-0001: Chunking Strategy

## Status
Accepted

## Context
Compliance documents include policy prose, regulatory bulletins, and audit narratives. Missing section boundaries can cause citation drift.

## Decision
Use hierarchical chunking:
- Level 1: document section boundaries
- Level 2: paragraph-aware split
- Level 3: fixed-size overlap windows (700 chars, overlap 120)

## Consequences
- Better citation precision to section-level IDs.
- Slight increase in index size from overlap.
- Supports future parent-child retrieval with section metadata.
