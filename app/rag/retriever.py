from __future__ import annotations

import re
from pathlib import Path

from app.core.config import settings
from app.domain.models import Chunk

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]{3,}\b", text.lower())


def _lexical_overlap_score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return float(len(query_tokens.intersection(chunk_tokens))) / float(len(query_tokens))


def _query_doc_type_hint(query: str) -> str | None:
    lower_query = query.lower()
    if "audit" in lower_query:
        return "audit"
    if "regulation" in lower_query or "regulatory" in lower_query:
        return "regulation"
    if "policy" in lower_query:
        return "policy"
    return None


def _source_boost(doc_id: str) -> float:
    # Prefer curated policy/audit/regulation sources over synthetic controls.
    return -0.08 if doc_id.startswith("SYN-") else 0.12


class VectorRetriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._chunk_tokens: list[set[str]] = []
        self._embedder = None
        self._chroma_client = None
        self._collection = None

        if not settings.enable_embedding_retriever:
            return

        sentence_transformer_cls = None
        try:
            from sentence_transformers import SentenceTransformer

            sentence_transformer_cls = SentenceTransformer
        except ImportError:  # pragma: no cover
            sentence_transformer_cls = None

        if chromadb is not None and sentence_transformer_cls is not None:
            try:
                persist_dir = Path(settings.chroma_persist_dir)
                persist_dir.mkdir(parents=True, exist_ok=True)

                self._embedder = sentence_transformer_cls(
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
        query_tokens = set(_tokenize(query))
        type_hint = _query_doc_type_hint(query)

        if self._collection is not None and self._embedder is not None:
            query_embedding = self._embedder.encode([query], normalize_embeddings=True).tolist()[0]

            # Fetch more candidates for optional jurisdiction filtering.
            query_k = max(top_k * 4, top_k)
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=query_k,
                include=["documents", "metadatas", "distances"],
            )

            vector_scores: dict[tuple[str, str], float] = {}
            vector_chunks: dict[tuple[str, str], Chunk] = {}
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
                key = (chunk.chunk_id, chunk.section_id)
                vector_scores[key] = score
                vector_chunks[key] = chunk

            hybrid_scores: list[tuple[Chunk, float]] = []
            for chunk, chunk_tokens in zip(self._chunks, self._chunk_tokens):
                if jurisdiction and chunk.jurisdiction not in {jurisdiction, "global"}:
                    continue

                key = (chunk.chunk_id, chunk.section_id)
                vector_score = vector_scores.get(key, 0.0)
                lexical_score = _lexical_overlap_score(query_tokens, chunk_tokens)
                type_boost = 0.06 if type_hint is not None and chunk.doc_type == type_hint else 0.0
                hybrid_score = 0.65 * vector_score + 0.35 * lexical_score + type_boost + _source_boost(chunk.doc_id)
                hybrid_scores.append((vector_chunks.get(key, chunk), hybrid_score))

            hybrid_scores.sort(key=lambda item: item[1], reverse=True)
            return hybrid_scores[:top_k]

        if not self._chunks:
            raise RuntimeError("Retriever index has not been built.")

        scored: list[tuple[Chunk, float]] = []
        for chunk, chunk_tokens in zip(self._chunks, self._chunk_tokens):
            if jurisdiction and chunk.jurisdiction != jurisdiction and chunk.jurisdiction != "global":
                continue
            overlap_score = _lexical_overlap_score(query_tokens, chunk_tokens)
            type_boost = 0.06 if type_hint is not None and chunk.doc_type == type_hint else 0.0
            score = overlap_score + type_boost + _source_boost(chunk.doc_id)
            scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
