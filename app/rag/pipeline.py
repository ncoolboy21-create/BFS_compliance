from __future__ import annotations

import json

from app.core.config import settings
from app.domain.models import QARequest, QAResponse, SourceDocument
from app.rag.chunking import hierarchical_chunk
from app.rag.generator import CitationAwareGenerator
from app.rag.reranker import Reranker
from app.rag.retriever import VectorRetriever
from app.services.escalation import escalation_reasons


class ComplianceRAGPipeline:
    def __init__(self) -> None:
        self._retriever = VectorRetriever()
        self._reranker = Reranker()
        self._generator = CitationAwareGenerator()
        self._is_ready = False
        self._indexed_docs: dict[str, SourceDocument] = {}

    def load_documents(self, data_path: str) -> int:
        documents: list[SourceDocument] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                doc = SourceDocument(**payload)
                documents.append(doc)
                self._indexed_docs[doc.doc_id] = doc

        chunks = hierarchical_chunk(
            documents,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
        self._retriever.build_index(chunks)
        self._is_ready = True
        return len(chunks)

    def ingest_document(self, doc: SourceDocument) -> int:
        """Add a single document to the live index without full reload."""
        new_chunks = hierarchical_chunk(
            [doc],
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
        self._retriever.add_chunks(new_chunks)
        self._indexed_docs[doc.doc_id] = doc
        self._is_ready = True
        return len(new_chunks)

    def list_documents(self) -> list[dict]:
        return [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "doc_type": d.doc_type,
                "jurisdiction": d.jurisdiction,
                "section_count": len(d.sections),
            }
            for d in self._indexed_docs.values()
        ]

    def ask(self, req: QARequest) -> QAResponse:
        if not self._is_ready:
            raise RuntimeError("Pipeline is not initialized.")

        retrieved = self._retriever.retrieve(
            query=req.question,
            top_k=settings.top_k_retrieval,
            jurisdiction=req.jurisdiction,
        )
        reranked = self._reranker.rerank(req.question, retrieved, top_k=settings.top_k_rerank)

        contexts = [
            {
                "doc_id": chunk.doc_id,
                "section_id": chunk.section_id,
                "text": chunk.text,
                "score": score,
            }
            for chunk, score in reranked
        ]
        answer, citations = self._generator.generate(req.question, contexts, req.max_citations)

        confidence = self._reranker.confidence_from_scores([score for _, score in reranked])
        reasons = escalation_reasons(req.question, confidence, bool(citations))
        trust_decision = "AI_AUTO_APPROVED"

        return QAResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            trust_decision=trust_decision,
            escalation_reasons=reasons,
            retrieval_debug={
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "top_docs": [chunk.doc_id for chunk, _ in reranked],
            },
        )
