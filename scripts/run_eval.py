from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from datasets import Dataset

from app.domain.models import QARequest
from app.rag.pipeline import ComplianceRAGPipeline

try:
    from ragas import evaluate
    from ragas.metrics import context_recall as ragas_context_recall_metric
    from ragas.metrics import faithfulness as ragas_faithfulness_metric
except Exception:  # pragma: no cover
    evaluate = None  # type: ignore
    ragas_context_recall_metric = None  # type: ignore
    ragas_faithfulness_metric = None  # type: ignore

GOLDEN_PATH = Path("data/golden_set.json")
OUT_JSON = Path("docs/ragas-scorecard.json")
OUT_MD = Path("docs/ragas-scorecard.md")


def lexical_overlap(a: str, b: str) -> float:
    a_terms = {t for t in a.lower().split() if len(t) > 2}
    b_terms = {t for t in b.lower().split() if len(t) > 2}
    if not a_terms:
        return 0.0
    return len(a_terms.intersection(b_terms)) / len(a_terms)


def main() -> None:
    pipeline = ComplianceRAGPipeline()
    pipeline.load_documents("data/synthetic_corpus.jsonl")

    rows = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    result_rows = []
    faithfulness_scores: list[float] = []
    recall_scores: list[float] = []
    ragas_rows: list[dict[str, object]] = []

    for row in rows:
        res = pipeline.ask(QARequest(question=row["question"], max_citations=4))
        got_citations = {f"{c.doc_id}#{c.section_id}" for c in res.citations}
        expected = set(row["expected_citations"])

        citation_recall = len(got_citations.intersection(expected)) / max(1, len(expected))
        answer_overlap = lexical_overlap(row["ground_truth"], res.answer)

        # Proxy faithfulness: answer lexical grounding + citation support.
        proxy_faithfulness = 0.6 * answer_overlap + 0.4 * citation_recall

        faithfulness_scores.append(proxy_faithfulness)
        recall_scores.append(citation_recall)

        ragas_rows.append(
            {
                "question": row["question"],
                "answer": res.answer,
                "contexts": [c.quote for c in res.citations] or [""],
                "ground_truth": row["ground_truth"],
            }
        )

        result_rows.append(
            {
                "question": row["question"],
                "faithfulness": round(proxy_faithfulness, 3),
                "context_recall": round(citation_recall, 3),
                "confidence": round(res.confidence, 3),
                "citations": sorted(got_citations),
            }
        )

    aggregate = {
        "faithfulness": round(mean(faithfulness_scores), 3),
        "context_recall": round(mean(recall_scores), 3),
        "pass_faithfulness": mean(faithfulness_scores) >= 0.90,
        "pass_context_recall": mean(recall_scores) >= 0.85,
        "scoring_mode": "proxy",
    }

    if (
        evaluate is not None
        and ragas_faithfulness_metric is not None
        and ragas_context_recall_metric is not None
    ):
        try:
            ds = Dataset.from_list(ragas_rows)
            ragas_result = evaluate(ds, metrics=[ragas_faithfulness_metric, ragas_context_recall_metric])
            ragas_df = ragas_result.to_pandas()
            ragas_faithfulness = float(ragas_df["faithfulness"].mean())
            ragas_context_recall = float(ragas_df["context_recall"].mean())

            aggregate = {
                "faithfulness": round(ragas_faithfulness, 3),
                "context_recall": round(ragas_context_recall, 3),
                "pass_faithfulness": ragas_faithfulness >= 0.90,
                "pass_context_recall": ragas_context_recall >= 0.85,
                "scoring_mode": "ragas",
            }
        except Exception:
            # Keep proxy results when RAGAS runtime prerequisites are missing.
            pass

    payload = {"aggregate": aggregate, "rows": result_rows}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# RAGAS Scorecard ({aggregate['scoring_mode']})",
        "",
        f"- Faithfulness: **{aggregate['faithfulness']:.3f}** (target >= 0.90)",
        f"- Context Recall: **{aggregate['context_recall']:.3f}** (target >= 0.85)",
        f"- Pass: **{aggregate['pass_faithfulness'] and aggregate['pass_context_recall']}**",
        "",
        "| Question | Faithfulness | Context Recall | Confidence |",
        "|---|---:|---:|---:|",
    ]
    for item in result_rows:
        md_lines.append(
            f"| {item['question']} | {item['faithfulness']:.3f} | {item['context_recall']:.3f} | {item['confidence']:.3f} |"
        )

    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
