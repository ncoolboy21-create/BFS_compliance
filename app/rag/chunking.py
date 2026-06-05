from __future__ import annotations

import re

from app.domain.models import Chunk, SourceDocument


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    segments: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))

        # Prefer splitting on whitespace so chunks do not start/end mid-word.
        if end < len(cleaned) and cleaned[end].isalnum():
            backtrack = cleaned.rfind(" ", start, end)
            if backtrack > start + int(chunk_size * 0.6):
                end = backtrack

        segment = cleaned[start:end].strip()
        if segment:
            segments.append(segment)

        if end == len(cleaned):
            break

        start = max(0, end - overlap)
        # If overlap start lands inside a word, advance to next boundary.
        if 0 < start < len(cleaned) and cleaned[start - 1].isalnum() and cleaned[start].isalnum():
            while start < len(cleaned) and cleaned[start].isalnum():
                start += 1
            while start < len(cleaned) and cleaned[start] == " ":
                start += 1
    return segments


def hierarchical_chunk(
    documents: list[SourceDocument], chunk_size: int = 700, overlap: int = 120
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for section in doc.sections:
            paragraphs = [p.strip() for p in re.split(r"\n\n+", section.text) if p.strip()]
            if not paragraphs:
                paragraphs = [section.text]
            paragraph_text = "\n\n".join(paragraphs)
            segment_texts = _split_with_overlap(paragraph_text, chunk_size, overlap)
            for idx, segment in enumerate(segment_texts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}:{section.section_id}:{idx}",
                        doc_id=doc.doc_id,
                        doc_type=doc.doc_type,
                        section_id=section.section_id,
                        jurisdiction=doc.jurisdiction,
                        text=segment,
                    )
                )
    return chunks
