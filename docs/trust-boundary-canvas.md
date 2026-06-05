# Trust Boundary Canvas

## Zone Definitions
- Zone A - System of Record: trusted policy/regulation/audit source documents and approved metadata.
- Zone B - Retrieval and Evidence Compute: chunking, embedding, indexing, retrieval, reranking, confidence scoring.
- Zone C - LLM Inference: prompt construction and model inference only.
- Zone D - Human Oversight and Decision: compliance reviewer validation and final acceptance/rejection.

## Capability Placement (Exactly One Zone per Capability)

| Capability | Zone | Why This Zone |
|---|---|---|
| Raw document custody | Zone A | Canonical source material must remain in system-of-record boundary |
| Document schema validation | Zone A | Validation is part of record integrity control |
| Chunk generation | Zone B | Compute transformation for search index preparation |
| Embedding generation | Zone B | Retrieval feature extraction stage |
| ChromaDB index persistence | Zone B | Search-serving storage for evidence retrieval |
| Query retrieval (`top_k`) | Zone B | Evidence candidate selection logic |
| Cross-encoder reranking | Zone B | Evidence relevance optimization |
| Confidence calculation | Zone B | Evidence-derived trust signal computation |
| Prompt assembly with contexts | Zone C | Input packaging for LLM inference |
| LLM answer synthesis | Zone C | Probabilistic text generation stage |
| Citation object parsing | Zone C | Model-output post-processing in inference boundary |
| Escalation reason tagging | Zone D | Human-governance policy interpretation point |
| Final approval/rejection | Zone D | Accountable decision authority |
| Audit sign-off | Zone D | Human attestation and compliance accountability |

## Boundary Interfaces and Controls

### Boundary A -> B (Record to Retrieval Compute)
- Controls:
  - File/type validation
  - Source metadata retention (`doc_id`, `section_id`, jurisdiction)
  - Ingestion logging with timestamps
- Threats addressed:
  - malformed content ingestion
  - source provenance loss

### Boundary B -> C (Evidence Compute to LLM Inference)
- Controls:
  - context-only prompt policy
  - citation-required output schema
  - capped context window and chunk count
- Threats addressed:
  - ungrounded generation
  - prompt contamination from irrelevant context

### Boundary C -> D (LLM Output to Human Decision)
- Controls:
  - confidence score visibility
  - escalation reason visibility
  - mandatory human review for threshold/risk conditions
- Threats addressed:
  - blind auto-acceptance
  - opaque recommendation usage

## Trust Rules
- Auto-approve only when confidence >= configured threshold and no mandatory risk gate is triggered.
- Require human approval for low-confidence, policy-boundary, jurisdiction-specific, or missing-citation responses.
- Maintain logs for provider failures and fallback mode activation.

## Out-of-Scope Capabilities
- No autonomous policy update execution.
- No autonomous external filing submission.
- No automatic legal interpretation sign-off.
