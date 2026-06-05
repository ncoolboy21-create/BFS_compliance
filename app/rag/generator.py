from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from typing import Any

from app.core.config import settings
from app.domain.models import Citation

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:  # pragma: no cover
    AzureOpenAI = None
    OpenAI = None


logger = logging.getLogger(__name__)


class CitationAwareGenerator:
    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = text.replace("\r", " ").replace("\n", " ")
        cleaned = cleaned.replace(" $ ", "; ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _trim_for_readability(text: str, max_chars: int) -> str:
        normalized = CitationAwareGenerator._normalize_text(text)
        if len(normalized) <= max_chars:
            return normalized

        clipped = normalized[:max_chars].rstrip()
        # Avoid chopping words mid-way.
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]

        return clipped + "..."

    @staticmethod
    def _clean_leading_fragment(text: str) -> str:
        cleaned = CitationAwareGenerator._normalize_text(text)
        if not cleaned:
            return cleaned

        tokens = cleaned.split()
        if len(tokens) > 1 and len(tokens[0]) <= 3 and tokens[0].islower():
            cleaned = " ".join(tokens[1:])

        tokens = cleaned.split()
        if len(tokens) > 1 and len(tokens[-1]) <= 3 and tokens[-1].islower():
            cleaned = " ".join(tokens[:-1]).rstrip(" ,;:")

        return cleaned

    @staticmethod
    def _merge_with_overlap(parts: list[str], max_overlap: int = 180) -> str:
        if not parts:
            return ""

        merged = parts[0]
        for part in parts[1:]:
            best = 0
            bound = min(max_overlap, len(merged), len(part))
            for k in range(bound, 19, -1):
                if merged[-k:] == part[:k]:
                    best = k
                    break
            if best == 0 and merged and part and not merged.endswith((" ", "\n")) and not part.startswith((" ", ",", ".", ";", ":")):
                merged += " "
            merged += part[best:]
        return merged

    @staticmethod
    def _section_sort_key(section_id: str) -> tuple[int, str]:
        m = re.search(r"(\d+)", section_id)
        if m:
            return (int(m.group(1)), section_id)
        return (10**9, section_id)

    @staticmethod
    def _chunk_sort_key(chunk_id: str | None) -> tuple[int, str]:
        if not chunk_id:
            return (10**9, "")

        m = re.search(r":(\d+)$", chunk_id)
        if m:
            return (int(m.group(1)), chunk_id)
        return (10**9, chunk_id)

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_terms = set(re.findall(r"\b[a-z0-9]{3,}\b", a.lower()))
        b_terms = set(re.findall(r"\b[a-z0-9]{3,}\b", b.lower()))
        if not a_terms or not b_terms:
            return 0.0
        inter = len(a_terms.intersection(b_terms))
        union = len(a_terms.union(b_terms))
        return inter / max(1, union)

    @staticmethod
    def _dedupe_entries(entries: list[tuple[str, str, str]], threshold: float = 0.82) -> list[tuple[str, str, str]]:
        unique: list[tuple[str, str, str]] = []
        for candidate in entries:
            _, _, candidate_text = candidate
            is_duplicate = False
            for _, _, existing_text in unique:
                if CitationAwareGenerator._text_similarity(candidate_text, existing_text) >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(candidate)
        return unique

    def __init__(self) -> None:
        self._use_openai = (
            settings.use_openai
            and settings.openai_api_key
            and settings.openai_model_name
        )
        self._use_azure = (
            settings.use_azure_openai
            and settings.azure_openai_api_key
            and settings.azure_openai_endpoint
            and settings.azure_openai_deployment
        )
        self._use_gemini = settings.use_gemini and bool(settings.gemini_api_key)

        self._model = None
        self._client = None

        if self._use_openai and OpenAI is not None:
            try:
                self._client = OpenAI(api_key=settings.openai_api_key)
                self._model = "openai"
            except Exception as exc:
                logger.warning("OpenAI initialization failed; trying next provider. Error: %s", exc)
                self._model = None
        elif self._use_azure and AzureOpenAI is not None:
            try:
                self._client = AzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                    azure_endpoint=settings.azure_openai_endpoint,
                )
                self._model = "azure_openai"
            except Exception as exc:
                logger.warning("Azure OpenAI initialization failed; using fallback generation. Error: %s", exc)
                self._model = None
        elif self._use_gemini and genai is not None:
            try:
                genai.configure(api_key=settings.gemini_api_key)
                self._model = genai.GenerativeModel(settings.gemini_model_name)
            except Exception as exc:
                logger.warning("Gemini initialization failed; using fallback generation. Error: %s", exc)
                self._model = None

    def build_prompt(self, question: str, contexts: list[dict[str, Any]], max_citations: int) -> str:
        context_block = "\n\n".join(
            [
                f"[Source: {ctx['doc_id']} - Section {ctx['section_id']}]\n{ctx['text']}"
                for ctx in contexts
            ]
        )
        return (
            f"Question: {question}\n\n"
            f"Context sources:\n{context_block}\n\n"
            f"Based on the context above, provide a comprehensive and well-structured answer. "
            f"Include relevant details and cite your sources. Use up to {max_citations} citations.\n\n"
            f"Return your response as JSON with this structure:\n"
            f'{{"answer": "detailed answer here", "citations": [{{"doc_id": "...", "section_id": "...", "quote": "..."}}]}}'
        )

    def generate(self, question: str, contexts: list[dict[str, Any]], max_citations: int) -> tuple[str, list[Citation]]:
        if self._model is not None:
            try:
                prompt = self.build_prompt(question, contexts, max_citations)
                
                if self._model == "openai" and self._client is not None:
                    response = self._client.chat.completions.create(
                        model=settings.openai_model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a professional compliance assistant. Provide detailed, well-structured answers based on the provided context. Always respond with valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=settings.openai_temperature,
                        response_format={"type": "json_object"},
                    )
                    parsed = self._safe_parse_json(response.choices[0].message.content or "")
                elif self._model == "azure_openai" and self._client is not None:
                    response = self._client.chat.completions.create(
                        model=settings.azure_openai_deployment,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a professional compliance assistant. Provide detailed, well-structured answers based on the provided context. Always respond with valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=settings.azure_openai_temperature,
                    )
                    parsed = self._safe_parse_json(response.choices[0].message.content or "")
                else:
                    response = self._model.generate_content(
                        prompt,
                        temperature=settings.gemini_temperature,
                    )
                    parsed = self._safe_parse_json(response.text)
                
                citations = [Citation(**item) for item in parsed.get("citations", [])[:max_citations]]
                answer = str(parsed.get("answer", "")).strip()
                if answer:
                    return answer, citations
                logger.warning("LLM response missing 'answer'; using fallback formatter.")
            except Exception as exc:
                logger.warning("LLM generation failed; using fallback formatter. Error: %s", exc)
                pass

        # Deterministic local fallback for development and evaluation.
        picked = contexts[:max_citations]
        unique_citations: list[Citation] = []
        seen_citation_keys: set[tuple[str, str]] = set()
        for item in picked:
            citation_key = (item["doc_id"], item["section_id"])
            if citation_key in seen_citation_keys:
                continue
            seen_citation_keys.add(citation_key)
            unique_citations.append(
                Citation(doc_id=item["doc_id"], section_id=item["section_id"], quote=item["text"][:200])
            )
        citations = unique_citations
        if not picked:
            return "Insufficient evidence in provided context.", []

        grouped: "OrderedDict[tuple[str, str], list[str]]" = OrderedDict()
        grouped_chunks: dict[tuple[str, str], list[tuple[tuple[int, str], str]]] = {}
        for item in picked:
            key = (item["doc_id"], item["section_id"])
            grouped.setdefault(key, []).append(self._clean_leading_fragment(item["text"]))
            grouped_chunks.setdefault(key, []).append(
                (
                    self._chunk_sort_key(str(item.get("chunk_id", ""))),
                    self._clean_leading_fragment(item["text"]),
                )
            )

        merged_entries: list[tuple[str, str, str]] = []
        for (doc_id, section_id), parts in grouped.items():
            ordered_parts = [text for _, text in sorted(grouped_chunks[(doc_id, section_id)], key=lambda pair: pair[0])]
            merged_text = self._merge_with_overlap(ordered_parts)
            merged_entries.append((doc_id, section_id, merged_text))

        merged_entries.sort(key=lambda item: (item[0], self._section_sort_key(item[1])))
        merged_entries = self._dedupe_entries(merged_entries)

        primary_text = merged_entries[0][2] if merged_entries else self._clean_leading_fragment(picked[0]["text"])
        answer_lines = [
            "Based on the compliance evidence reviewed:",
            self._trim_for_readability(primary_text, 5000),
        ]
        if len(merged_entries) > 1:
            answer_lines.append("\nAdditional relevant details:")
            for _, _, text in merged_entries[1:]:
                answer_lines.append(f"- {self._trim_for_readability(text, 1500)}")

        source_summary = ", ".join(
            [f"{c.doc_id} (Section {c.section_id})" for c in citations]
        )
        if source_summary:
            answer_lines.append(f"\nSources reviewed: {source_summary}")
        return "\n".join(answer_lines), citations

    @staticmethod
    def _safe_parse_json(payload: str) -> dict[str, Any]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"answer": payload, "citations": []}
