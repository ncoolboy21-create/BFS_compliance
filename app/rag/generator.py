from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.domain.models import Citation

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


class CitationAwareGenerator:
    def __init__(self) -> None:
        self._use_gemini = settings.use_gemini and bool(settings.gemini_api_key)
        if self._use_gemini and genai is not None:
            genai.configure(api_key=settings.gemini_api_key)
            self._model = genai.GenerativeModel(settings.gemini_model_name)
        else:
            self._model = None

    def build_prompt(self, question: str, contexts: list[dict[str, Any]], max_citations: int) -> str:
        context_block = "\n\n".join(
            [
                f"[DOC={ctx['doc_id']} SEC={ctx['section_id']}]\n{ctx['text']}"
                for ctx in contexts
            ]
        )
        return (
            "You are a compliance assistant. Answer only from provided context. "
            "If evidence is insufficient, explicitly say so. "
            "Return JSON with keys: answer, citations. "
            "citations must be a list of objects {doc_id, section_id, quote}. "
            f"Use up to {max_citations} citations.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_block}"
        )

    def generate(self, question: str, contexts: list[dict[str, Any]], max_citations: int) -> tuple[str, list[Citation]]:
        if self._model is not None:
            try:
                prompt = self.build_prompt(question, contexts, max_citations)
                response = self._model.generate_content(prompt)
                parsed = self._safe_parse_json(response.text)
                citations = [Citation(**item) for item in parsed.get("citations", [])[:max_citations]]
                return parsed.get("answer", "Insufficient evidence in provided context."), citations
            except Exception:
                # If model access fails (auth/model/network), continue with deterministic local fallback.
                pass

        # Deterministic local fallback for development and evaluation.
        picked = contexts[:max_citations]
        citations = [
            Citation(doc_id=item["doc_id"], section_id=item["section_id"], quote=item["text"][:200])
            for item in picked
        ]
        if not picked:
            return "Insufficient evidence in provided context.", []

        answer_lines = [
            "AI recommendation (human approval required):",
            f"Based on retrieved evidence, {picked[0]['text'][:260].strip()}...",
        ]
        for item in picked:
            answer_lines.append(f"[{item['doc_id']}#{item['section_id']}]")
        return "\n".join(answer_lines), citations

    @staticmethod
    def _safe_parse_json(payload: str) -> dict[str, Any]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"answer": payload, "citations": []}
