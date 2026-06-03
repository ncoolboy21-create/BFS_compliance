# ADR-0004: Generation and Trust Control

## Status
Accepted

## Context
Wrong answers carry regulatory risk; the architecture must enforce review gates.

## Decision
- Prompt requires citations for every answer.
- Trust mode fixed to `AI_RECOMMEND_HUMAN_APPROVE`.
- Escalate on low confidence, jurisdiction-specific wording, policy-boundary wording, or missing citations.

## Consequences
- Reduced autonomous throughput.
- Increased explainability and auditability.
- Explicit operational path for high-risk questions.
