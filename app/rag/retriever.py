from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.domain.models import Chunk


class VectorRetriever:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus = [chunk.text for chunk in chunks]
        matrix = self._vectorizer.fit_transform(corpus)
        self._matrix = matrix.toarray()

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        """Incrementally add chunks and rebuild the TF-IDF index."""
        self._chunks.extend(new_chunks)
        corpus = [chunk.text for chunk in self._chunks]
        matrix = self._vectorizer.fit_transform(corpus)
        self._matrix = matrix.toarray()

    def retrieve(self, query: str, top_k: int = 12, jurisdiction: str | None = None) -> list[tuple[Chunk, float]]:
        if self._matrix is None:
            raise RuntimeError("Retriever index has not been built.")

        q_vec = self._vectorizer.transform([query]).toarray()[0]
        q_norm = np.linalg.norm(q_vec) + 1e-12
        d_norm = np.linalg.norm(self._matrix, axis=1) + 1e-12
        sims = (self._matrix @ q_vec) / (d_norm * q_norm)

        ranked_idx = np.argsort(sims)[::-1]
        ranked: list[tuple[Chunk, float]] = []
        for idx in ranked_idx:
            chunk = self._chunks[int(idx)]
            if jurisdiction and chunk.jurisdiction != jurisdiction and chunk.jurisdiction != "global":
                continue
            ranked.append((chunk, float(sims[int(idx)])))
            if len(ranked) >= top_k:
                break
        return ranked
