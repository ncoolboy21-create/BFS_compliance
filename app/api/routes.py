from __future__ import annotations

import json
import re
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.domain.models import QARequest, QAResponse, Section, SourceDocument
from app.rag.cost_latency import estimate_cost_latency
from app.rag.pipeline import ComplianceRAGPipeline


def _pdf_to_source_doc(content: bytes, filename: str) -> SourceDocument:
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(content))
        sections: list[Section] = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(Section(
                    section_id=f"page-{i + 1}",
                    title=f"Page {i + 1}",
                    text=text,
                ))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PDF parse error: {exc}") from exc

    if not sections:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    doc_id = re.sub(r"[^a-z0-9_-]", "_", filename.lower().removesuffix(".pdf"))
    return SourceDocument(
        doc_id=doc_id or str(uuid.uuid4()),
        doc_type="policy",
        title=filename.removesuffix(".pdf").replace("_", " ").title(),
        jurisdiction="global",
        sections=sections,
    )


def _txt_to_source_doc(content: bytes, filename: str) -> SourceDocument:
    text = content.decode("utf-8", errors="replace").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Uploaded text file is empty.")
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    sections = [
        Section(section_id=f"para-{i + 1}", title=f"Paragraph {i + 1}", text=p)
        for i, p in enumerate(paragraphs)
    ] or [Section(section_id="body", title="Body", text=text)]
    doc_id = re.sub(r"[^a-z0-9_-]", "_", filename.lower().removesuffix(".txt"))
    return SourceDocument(
        doc_id=doc_id or str(uuid.uuid4()),
        doc_type="policy",
        title=filename.removesuffix(".txt").replace("_", " ").title(),
        jurisdiction="global",
        sections=sections,
    )


def build_router(pipeline: ComplianceRAGPipeline) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/ask", response_model=QAResponse)
    def ask(req: QARequest) -> QAResponse:
        try:
            return pipeline.ask(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/estimate")
    def estimate(req: QARequest) -> dict[str, object]:
        estimate_obj = estimate_cost_latency(req.question, context_chars=4500)
        return {
            "input_tokens": estimate_obj.input_tokens,
            "output_tokens": estimate_obj.output_tokens,
            "estimated_cost_usd": estimate_obj.estimated_cost_usd,
            "estimated_latency_ms": estimate_obj.estimated_latency_ms,
            "within_rpm_limit": estimate_obj.within_rpm_limit,
            "within_tpm_limit": estimate_obj.within_tpm_limit,
        }

    @router.get("/documents")
    def list_documents() -> dict[str, object]:
        return {"documents": pipeline.list_documents()}

    @router.post("/upload")
    async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
        filename = file.filename or "uploaded_file"
        content = await file.read()
        ext = filename.rsplit(".", 1)[-1].lower()

        if ext == "pdf":
            doc = _pdf_to_source_doc(content, filename)
        elif ext == "txt":
            doc = _txt_to_source_doc(content, filename)
        elif ext in ("json", "jsonl"):
            try:
                text = content.decode("utf-8", errors="replace")
                # Try single-doc JSON first, then JSONL, then array
                if ext == "json":
                    raw = json.loads(text)
                    docs_raw = raw if isinstance(raw, list) else [raw]
                else:
                    docs_raw = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"JSON parse error: {exc}") from exc

            total_chunks = 0
            ingested = []
            for raw_doc in docs_raw:
                try:
                    doc_obj = SourceDocument(**raw_doc)
                    n = pipeline.ingest_document(doc_obj)
                    total_chunks += n
                    ingested.append(doc_obj.doc_id)
                except Exception as exc:
                    raise HTTPException(status_code=422, detail=f"Document validation error: {exc}") from exc

            return {
                "status": "ingested",
                "doc_ids": ingested,
                "chunks_added": total_chunks,
            }
        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Upload PDF, TXT, JSON, or JSONL.",
            )

        chunks_added = pipeline.ingest_document(doc)
        return {
            "status": "ingested",
            "doc_ids": [doc.doc_id],
            "chunks_added": chunks_added,
        }

    return router
