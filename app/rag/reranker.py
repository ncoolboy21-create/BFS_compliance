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
    def confidence_from_scores(scores: list[float]) -> float:
        if not scores:
            return 0.0
        mean_score = sum(scores) / len(scores)
        return 1 / (1 + math.exp(-mean_score))
