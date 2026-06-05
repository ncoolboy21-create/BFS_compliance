from __future__ import annotations

import math

from app.core.config import settings
from app.domain.models import Chunk

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self._model = None
        model_to_use = model_name or settings.reranker_model_name
        if CrossEncoder is not None:
            try:
                self._model = CrossEncoder(
                    model_to_use,
                    local_files_only=settings.reranker_local_files_only,
                )
            except Exception:
                self._model = None

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []

        if self._model is not None:
            pairs = [(query, chunk.text) for chunk, _ in candidates]
            raw_scores = self._model.predict(pairs)
            score_pairs = [(candidates[i][0], float(raw_scores[i])) for i in range(len(candidates))]
            score_pairs.sort(key=lambda item: item[1], reverse=True)
            return score_pairs[:top_k]

        # Fallback lexical reranking if model download/runtime is unavailable.
        query_terms = {token for token in query.lower().split() if len(token) > 2}
        scored: list[tuple[Chunk, float]] = []
        for chunk, base_score in candidates:
            text_terms = set(chunk.text.lower().split())
            overlap = len(query_terms.intersection(text_terms))
            normalized_overlap = overlap / max(1, len(query_terms))
            score = 0.6 * float(base_score) + 0.4 * normalized_overlap
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def confidence_from_scores(scores: list[float], citation_count: int = 0) -> float:
        if not scores:
            return 0.0

        min_score = min(scores)
        max_score = max(scores)
        
        if max_score <= min_score:
            normalized_scores = [0.5] * len(scores)
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        mean_score = sum(normalized_scores) / len(normalized_scores)
        top_score = max(normalized_scores)

        consistency = 1.0
        if len(normalized_scores) > 1:
            variance = sum((score - mean_score) ** 2 for score in normalized_scores) / (len(normalized_scores) - 1)
            consistency = max(0.0, 1.0 - 2.0 * math.sqrt(variance))

        confidence = 0.30 + 0.50 * (0.6 * mean_score + 0.4 * top_score)
        confidence += 0.12 * consistency
        confidence += min(0.08, 0.04 * citation_count)
        return min(1.0, max(0.0, confidence))
