from app.domain.models import QARequest
from app.rag.pipeline import ComplianceRAGPipeline


def test_pipeline_returns_citations() -> None:
    pipeline = ComplianceRAGPipeline()
    pipeline.load_documents("data/synthetic_corpus.jsonl")

    response = pipeline.ask(QARequest(question="What is required before account activation?"))

    assert response.citations
    assert response.trust_decision == "AI_RECOMMEND_HUMAN_APPROVE"
