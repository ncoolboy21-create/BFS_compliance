from __future__ import annotations

from app.core.config import settings


def escalation_reasons(question: str, confidence: float, has_citations: bool) -> list[str]:
    reasons: list[str] = []
    q = question.lower()

    if confidence < settings.low_confidence_threshold:
        reasons.append("low-confidence")

    jurisdiction_markers = ["cross-border", "eu", "uk", "state-specific", "jurisdiction"]
    if any(marker in q for marker in jurisdiction_markers):
        reasons.append("jurisdiction-specific")

    boundary_markers = ["legal interpretation", "override policy", "exception approval"]
    if any(marker in q for marker in boundary_markers):
        reasons.append("policy-boundary")

    if not has_citations:
        reasons.append("missing-citations")

    return reasons
