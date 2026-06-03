# Failure Mode Register

| # | Failure Mode | Cause | Effect | Detection | Mitigation | Owner | Severity |
|---:|---|---|---|---|---|---|---|
| 1 | Regulatory misstatement in answer | Hallucination or weak context | Regulatory breach risk | Citation mismatch review, QA spot checks | Block auto-action, human approval mandatory, prompt hardening | Compliance Lead | Critical |
| 2 | Missing citation in response | Prompt or parser failure | Unverifiable guidance | API response schema validation | Reject response, trigger escalation | Platform Eng | High |
| 3 | Wrong jurisdiction applied | Retrieval did not filter by jurisdiction | Non-compliant regional guidance | Jurisdiction tag checks | Enforce jurisdiction filter and escalation | Compliance Ops | High |
| 4 | Outdated policy cited | Corpus not refreshed | Stale controls applied | Document timestamp drift alerts | Scheduled ingestion and deprecation tagging | Data Ops | High |
| 5 | Reranker failure fallback overused | Model download/runtime issue | Reduced precision | Reranker health telemetry | Cache model, warm start, alert on fallback rate | MLOps | Medium |
| 6 | Low confidence ignored | UX or process bypass | Unsafe decisioning | Workflow audit logs | Hard gate human approval for low confidence | Compliance Ops | High |
| 7 | Prompt injection in source text | Malicious or malformed document | Instruction override | Input sanitizer + pattern scan | Strip unsafe directives, isolate retrieval text | Security | High |
| 8 | Incomplete audit trail | Logging gaps | Auditor cannot reconstruct decision | Audit log completeness monitor | Immutable logs with request and citation IDs | Platform Eng | Medium |
