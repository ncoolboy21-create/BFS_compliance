from __future__ import annotations

import re
from pathlib import Path

from app.core.config import settings
from app.domain.models import Chunk

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]{3,}\b", text.lower())


class VectorRetriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._chunk_tokens: list[set[str]] = []
        self._embedder = None
        self._chroma_client = None
        self._collection = None

        if chromadb is not None and SentenceTransformer is not None:
            try:
                persist_dir = Path(settings.chroma_persist_dir)
                persist_dir.mkdir(parents=True, exist_ok=True)

                self._embedder = SentenceTransformer(
                    settings.embedding_model_name,
                    local_files_only=settings.embedding_local_files_only,
                )
                self._chroma_client = chromadb.PersistentClient(path=str(persist_dir))
                self._collection = self._chroma_client.get_or_create_collection(
                    name=settings.chroma_collection_name
                )
            except Exception:
                # Fall back to lexical retrieval when embeddings or Chroma are unavailable.
                self._embedder = None
                self._chroma_client = None
                self._collection = None

    def _upsert_chunks(self, chunks: list[Chunk]) -> None:
        if self._collection is None or self._embedder is None or not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True).tolist()
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "doc_type": chunk.doc_type,
                "section_id": chunk.section_id,
                "jurisdiction": chunk.jurisdiction,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def build_index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._chunk_tokens = [set(_tokenize(chunk.text)) for chunk in chunks]
        self._upsert_chunks(chunks)

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        """Incrementally add chunks and rebuild the lightweight token index."""
        self._chunks.extend(new_chunks)
        self._chunk_tokens.extend([set(_tokenize(chunk.text)) for chunk in new_chunks])
        self._upsert_chunks(new_chunks)

    def retrieve(self, query: str, top_k: int = 12, jurisdiction: str | None = None) -> list[tuple[Chunk, float]]:
        if self._collection is not None and self._embedder is not None:
            query_embedding = self._embedder.encode([query], normalize_embeddings=True).tolist()[0]

            # Fetch more candidates for optional jurisdiction filtering.
            query_k = max(top_k * 4, top_k)
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=query_k,
                include=["documents", "metadatas", "distances"],
            )

            scored: list[tuple[Chunk, float]] = []
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            for doc_text, metadata, distance in zip(documents, metadatas, distances):
                if not metadata:
                    continue

                chunk_jurisdiction = str(metadata.get("jurisdiction", "global"))
                if jurisdiction and chunk_jurisdiction not in {jurisdiction, "global"}:
                    continue

                chunk = Chunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    doc_id=str(metadata.get("doc_id", "")),
                    doc_type=str(metadata.get("doc_type", "policy")),
                    section_id=str(metadata.get("section_id", "")),
                    jurisdiction=chunk_jurisdiction,
                    text=doc_text,
                )
                score = 1.0 / (1.0 + float(distance))
                scored.append((chunk, score))

            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_k]

        if not self._chunks:
            raise RuntimeError("Retriever index has not been built.")

        query_tokens = set(_tokenize(query))
        scored: list[tuple[Chunk, float]] = []
        for chunk, chunk_tokens in zip(self._chunks, self._chunk_tokens):
            if jurisdiction and chunk.jurisdiction != jurisdiction and chunk.jurisdiction != "global":
                continue
            overlap = len(query_tokens.intersection(chunk_tokens))
            score = float(overlap) / max(1, len(query_tokens))
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
