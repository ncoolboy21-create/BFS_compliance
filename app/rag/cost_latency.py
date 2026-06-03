from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


@dataclass
class CostLatencyEstimate:
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    estimated_latency_ms: int
    within_rpm_limit: bool
    within_tpm_limit: bool


def estimate_cost_latency(question: str, context_chars: int, expected_output_tokens: int = 220) -> CostLatencyEstimate:
    # Coarse approximation: 1 token ~= 4 chars for English policy text.
    input_tokens = max(1, (len(question) + context_chars) // 4)
    output_tokens = expected_output_tokens

    input_cost = (input_tokens / 1_000_000) * settings.gemini_input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * settings.gemini_output_cost_per_million
    estimated_cost_usd = round(input_cost + output_cost, 6)

    # Latency model: fixed overhead + linear token processing term.
    estimated_latency_ms = int(450 + 0.22 * input_tokens + 0.35 * output_tokens)

    within_rpm = True
    within_tpm = (input_tokens + output_tokens) <= settings.gemini_tpm_limit

    return CostLatencyEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost_usd,
        estimated_latency_ms=estimated_latency_ms,
        within_rpm_limit=within_rpm,
        within_tpm_limit=within_tpm,
    )
