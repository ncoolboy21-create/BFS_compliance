from __future__ import annotations

import argparse
import json
import re
from math import sqrt
from pathlib import Path
from statistics import mean, median, pstdev

from app.domain.models import QARequest
from app.rag.pipeline import ComplianceRAGPipeline
from app.rag.reranker import Reranker


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9]{3,}\b", text.lower()))


def lexical_overlap(a: str, b: str) -> float:
    a_terms = _tokenize(a)
    b_terms = _tokenize(b)
    if not a_terms:
        return 0.0
    return len(a_terms.intersection(b_terms)) / len(a_terms)


def load_golden_set(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(data_path: str, reranker_model: str | None = None) -> ComplianceRAGPipeline:
    pipeline = ComplianceRAGPipeline()
    pipeline.load_documents(data_path)
    if reranker_model is not None:
        pipeline._reranker = Reranker(model_name=reranker_model)
    return pipeline


def evaluate_pipeline(
    name: str,
    pipeline: ComplianceRAGPipeline,
    golden_rows: list[dict[str, object]],
    low_confidence_threshold: float,
    max_citations: int,
) -> dict[str, object]:
    confidences: list[float] = []
    citation_recalls: list[float] = []
    faithfulness_scores: list[float] = []
    citation_counts: list[int] = []

    for row in golden_rows:
        request = QARequest(
            question=row["question"],
            jurisdiction=row.get("jurisdiction"),
            max_citations=max_citations,
        )
        response = pipeline.ask(request)

        got_citations = {f"{c.doc_id}#{c.section_id}" for c in response.citations}
        expected = set(row.get("expected_citations", []))
        citation_recall = len(got_citations.intersection(expected)) / max(1, len(expected))
        answer_overlap = lexical_overlap(row.get("ground_truth", ""), response.answer)
        proxy_faithfulness = 0.6 * answer_overlap + 0.4 * citation_recall

        confidences.append(response.confidence)
        citation_recalls.append(citation_recall)
        faithfulness_scores.append(proxy_faithfulness)
        citation_counts.append(len(response.citations))

    low_confidence_rate = sum(1 for c in confidences if c < low_confidence_threshold) / max(1, len(confidences))
    result = {
        "model_name": name,
        "mean_confidence": round(mean(confidences), 4),
        "median_confidence": round(median(confidences), 4),
        "confidence_std": round(pstdev(confidences) if len(confidences) > 1 else 0.0, 4),
        "low_confidence_rate": round(low_confidence_rate, 4),
        "mean_citation_count": round(mean(citation_counts), 4),
        "mean_citation_recall": round(mean(citation_recalls), 4),
        "mean_proxy_faithfulness": round(mean(faithfulness_scores), 4),
        "row_count": len(confidences),
    }
    return result


def write_results(output_json: Path, output_md: Path, baseline: dict[str, object], tuned: dict[str, object]) -> None:
    payload = {
        "baseline": baseline,
        "tuned": tuned,
        "delta": {
            "mean_confidence": round(tuned["mean_confidence"] - baseline["mean_confidence"], 4),
            "median_confidence": round(tuned["median_confidence"] - baseline["median_confidence"], 4),
            "low_confidence_rate": round(tuned["low_confidence_rate"] - baseline["low_confidence_rate"], 4),
            "mean_citation_recall": round(tuned["mean_citation_recall"] - baseline["mean_citation_recall"], 4),
            "mean_proxy_faithfulness": round(tuned["mean_proxy_faithfulness"] - baseline["mean_proxy_faithfulness"], 4),
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Confidence Benchmark",
        "",
        "## Summary",
        "",
        f"- Baseline model: **{baseline['model_name']}**",
        f"- Tuned model: **{tuned['model_name']}**",
        "",
        "## Metrics",
        "",
        "| Model | Mean Conf | Median Conf | Std Dev | Low Conf Rate | Mean Citations | Recall | Faithfulness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {baseline['model_name']} | {baseline['mean_confidence']:.4f} | {baseline['median_confidence']:.4f} | "
            f"{baseline['confidence_std']:.4f} | {baseline['low_confidence_rate']:.4f} | {baseline['mean_citation_count']:.4f} | "
            f"{baseline['mean_citation_recall']:.4f} | {baseline['mean_proxy_faithfulness']:.4f} |"
        ),
        (
            f"| {tuned['model_name']} | {tuned['mean_confidence']:.4f} | {tuned['median_confidence']:.4f} | "
            f"{tuned['confidence_std']:.4f} | {tuned['low_confidence_rate']:.4f} | {tuned['mean_citation_count']:.4f} | "
            f"{tuned['mean_citation_recall']:.4f} | {tuned['mean_proxy_faithfulness']:.4f} |"
        ),
        "",
        "## Delta (Tuned - Baseline)",
        "",
        f"- Mean confidence delta: **{payload['delta']['mean_confidence']:+.4f}**",
        f"- Median confidence delta: **{payload['delta']['median_confidence']:+.4f}**",
        f"- Low confidence rate delta: **{payload['delta']['low_confidence_rate']:+.4f}**",
        f"- Recall delta: **{payload['delta']['mean_citation_recall']:+.4f}**",
        f"- Faithfulness delta: **{payload['delta']['mean_proxy_faithfulness']:+.4f}**",
    ]
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark confidence before and after reranker fine-tuning")
    parser.add_argument(
        "--baseline-model",
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Baseline reranker model name or local path",
    )
    parser.add_argument(
        "--tuned-model",
        default="models/reranker_lora_merged",
        help="Tuned reranker model directory or path",
    )
    parser.add_argument("--golden-path", default="data/golden_set.json")
    parser.add_argument("--corpus-path", default="data/synthetic_corpus.jsonl")
    parser.add_argument("--output-json", default="docs/confidence-benchmark.json")
    parser.add_argument("--output-md", default="docs/confidence-benchmark.md")
    parser.add_argument("--max-citations", type=int, default=4)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    golden_rows = load_golden_set(Path(args.golden_path))

    print("Loading baseline pipeline...")
    baseline_pipeline = build_pipeline(args.corpus_path, reranker_model=args.baseline_model)
    baseline_metrics = evaluate_pipeline(
        name=args.baseline_model,
        pipeline=baseline_pipeline,
        golden_rows=golden_rows,
        low_confidence_threshold=args.low_confidence_threshold,
        max_citations=args.max_citations,
    )

    print("Loading tuned pipeline...")
    tuned_pipeline = build_pipeline(args.corpus_path, reranker_model=args.tuned_model)
    tuned_metrics = evaluate_pipeline(
        name=args.tuned_model,
        pipeline=tuned_pipeline,
        golden_rows=golden_rows,
        low_confidence_threshold=args.low_confidence_threshold,
        max_citations=args.max_citations,
    )

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    write_results(output_json, output_md, baseline_metrics, tuned_metrics)

    print(json.dumps({"baseline": baseline_metrics, "tuned": tuned_metrics}, indent=2))
    print(f"Benchmark results written to {output_json} and {output_md}")


if __name__ == "__main__":
    main()
