from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DocType = Literal["policy", "audit", "regulation"]


class Section(BaseModel):
    section_id: str
    title: str
    text: str


class SourceDocument(BaseModel):
    doc_id: str
    doc_type: DocType
    title: str
    jurisdiction: str
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: DocType
    section_id: str
    jurisdiction: str
    text: str


class Citation(BaseModel):
    doc_id: str
    section_id: str
    quote: str


class QARequest(BaseModel):
    question: str = Field(min_length=5)
    jurisdiction: str | None = None
    max_citations: int = Field(default=4, ge=1, le=10)


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    faithfulness: float
    context_recall: float
    trust_decision: str
    escalation_reasons: list[str]
    retrieval_debug: dict[str, object]


class EvaluationRow(BaseModel):
    question: str
    ground_truth: str
    expected_citations: list[str]
