from __future__ import annotations

import json
import re

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

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b[a-z0-9]{3,}\b", text.lower()))

    @classmethod
    def _answer_grounding_overlap(cls, answer: str, grounded_context: str) -> float:
        answer_terms = cls._tokenize(answer)
        context_terms = cls._tokenize(grounded_context)
        if not answer_terms:
            return 0.0
        return len(answer_terms.intersection(context_terms)) / len(answer_terms)

    @staticmethod
    def _citation_recall(
        reranked: list[tuple[Chunk, float]],
        citations: list,
        top_k: int,
    ) -> float:
        if not reranked:
            return 0.0

        relevant = {
            (chunk.doc_id, chunk.section_id)
            for chunk, _ in reranked[:max(1, top_k)]
        }
        if not relevant:
            return 0.0

        cited = {(c.doc_id, c.section_id) for c in citations}
        return len(relevant.intersection(cited)) / len(relevant)

    def _append_adjacent_sections(
        self,
        contexts: list[dict[str, object]],
        reranked: list[tuple[Chunk, float]],
        max_docs: int = 2,
    ) -> list[dict[str, object]]:
        seen_pairs = {(str(c["doc_id"]), str(c["section_id"])) for c in contexts}
        expanded = list(contexts)

        for chunk, score in reranked[:max_docs]:
            doc = self._indexed_docs.get(chunk.doc_id)
            if doc is None:
                continue

            section_ids = [section.section_id for section in doc.sections]
            if chunk.section_id not in section_ids:
                continue

            idx = section_ids.index(chunk.section_id)
            adjacent_idx = idx + 1
            if adjacent_idx >= len(doc.sections):
                continue

            adjacent_section = doc.sections[adjacent_idx]
            pair = (doc.doc_id, adjacent_section.section_id)
            if pair in seen_pairs:
                continue

            expanded.append(
                {
                    "doc_id": doc.doc_id,
                    "section_id": adjacent_section.section_id,
                    "chunk_id": f"{doc.doc_id}:{adjacent_section.section_id}:adjacent",
                    "text": adjacent_section.text,
                    "score": float(score) * 0.95,
                }
            )
            seen_pairs.add(pair)

        return expanded

    def _authoritative_seed_contexts(self, question: str, max_items: int = 2) -> list[dict[str, object]]:
        lower_question = question.lower()

        # Keep synthetic-intent questions aligned with reranked synthetic contexts.
        if "synthetic" in lower_question:
            return []

        prefix: str | None = None
        if "audit" in lower_question:
            prefix = "AUD-"
        elif "policy" in lower_question:
            prefix = "POL-"
        elif "regulation" in lower_question or "regulatory" in lower_question:
            prefix = "REG-"

        if prefix is None:
            return []

        q_terms = self._tokenize(question)
        if not q_terms:
            return []

        ranked_sections: list[tuple[float, SourceDocument, object]] = []
        for doc in self._indexed_docs.values():
            if not doc.doc_id.startswith(prefix):
                continue

            for section in doc.sections:
                section_terms = self._tokenize(section.text)
                overlap = len(q_terms.intersection(section_terms)) / max(1, len(q_terms))
                if overlap <= 0:
                    continue
                ranked_sections.append((overlap, doc, section))

        ranked_sections.sort(key=lambda item: item[0], reverse=True)

        seeds: list[dict[str, object]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for overlap, doc, section in ranked_sections:
            pair = (doc.doc_id, section.section_id)
            if pair in seen_pairs:
                continue

            seeds.append(
                {
                    "doc_id": doc.doc_id,
                    "section_id": section.section_id,
                    "chunk_id": f"{doc.doc_id}:{section.section_id}:seed",
                    "text": section.text,
                    "score": overlap + 1.0,
                }
            )
            seen_pairs.add(pair)

            section_ids = [s.section_id for s in doc.sections]
            if section.section_id in section_ids:
                idx = section_ids.index(section.section_id)
                adjacent_idx = idx + 1
                if adjacent_idx < len(doc.sections) and len(seeds) < max_items:
                    adjacent = doc.sections[adjacent_idx]
                    adjacent_pair = (doc.doc_id, adjacent.section_id)
                    if adjacent_pair not in seen_pairs:
                        seeds.append(
                            {
                                "doc_id": doc.doc_id,
                                "section_id": adjacent.section_id,
                                "chunk_id": f"{doc.doc_id}:{adjacent.section_id}:seed-adjacent",
                                "text": adjacent.text,
                                "score": overlap + 0.9,
                            }
                        )
                        seen_pairs.add(adjacent_pair)

            if len(seeds) >= max_items:
                break

        return seeds

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
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": score,
            }
            for chunk, score in reranked
        ]
        seed_contexts = self._authoritative_seed_contexts(req.question, max_items=req.max_citations)
        if seed_contexts:
            seen_pairs = {(str(c["doc_id"]), str(c["section_id"])) for c in seed_contexts}
            contexts = seed_contexts + [
                c for c in contexts if (str(c["doc_id"]), str(c["section_id"])) not in seen_pairs
            ]
        contexts = self._append_adjacent_sections(contexts, reranked)
        answer, citations = self._generator.generate(req.question, contexts, req.max_citations)

        effective_recall_k = min(settings.top_k_rerank, req.max_citations)
        citation_recall = self._citation_recall(reranked, citations, effective_recall_k)

        cited_pairs = {(c.doc_id, c.section_id) for c in citations}
        cited_context_text = " ".join(
            c["text"]
            for c in contexts
            if (c["doc_id"], c["section_id"]) in cited_pairs
        )
        if not cited_context_text and contexts:
            cited_context_text = contexts[0]["text"]

        answer_overlap = self._answer_grounding_overlap(answer, cited_context_text)
        # Citation grounding is weighted higher than lexical phrasing similarity.
        faithfulness = 0.2 * answer_overlap + 0.8 * citation_recall

        reranker_confidence = self._reranker.confidence_from_scores(
            [score for _, score in reranked],
            citation_count=len(citations),
        )
        grounding_confidence = 0.6 * faithfulness + 0.4 * citation_recall
        confidence = min(1.0, max(0.0, 0.7 * reranker_confidence + 0.3 * grounding_confidence))
        reasons = escalation_reasons(req.question, confidence, bool(citations))
        trust_decision = "AI_AUTO_APPROVED" if confidence >= settings.auto_approval_confidence_threshold else "HUMAN_APPROVAL_REQUIRED"

        return QAResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            faithfulness=faithfulness,
            context_recall=citation_recall,
            trust_decision=trust_decision,
            escalation_reasons=reasons,
            retrieval_debug={
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "top_docs": [chunk.doc_id for chunk, _ in reranked],
                "scoring_mode": "proxy_runtime",
            },
        )
