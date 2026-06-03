# Regional Bank Compliance RAG (FastAPI + DDD)

This project implements a citation-first RAG assistant for compliance officers over a synthetic 20-document corpus.

## Scope Implemented
- FastAPI API layer
- DDD bounded contexts: Policy, Audit, Regulation
- RAG pipeline with hierarchical chunking and required reranker
- Citation-required answer generation
- Trust architecture: `AI_RECOMMEND_HUMAN_APPROVE`
- Golden set evaluation and scorecard generation

## Project Structure
- `app/` API and RAG implementation
- `data/` synthetic corpus and golden set
- `docs/adr/` ADR pack for chunking, embedding/vector store, reranker, trust controls
- `docs/trust-boundary-canvas.md` trust boundary canvas
- `docs/failure-mode-register.md` FMEA register (8+ failure modes)
- `scripts/run_eval.py` scorecard generator

## Quick Start
1. Create and activate a Python 3.11+ environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run API:
   - `uvicorn app.main:app --reload`
4. Open Swagger:
   - `http://127.0.0.1:8000/docs`

## Example API Call
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is required before customer onboarding account activation?","jurisdiction":"global"}'
```

## Evaluation
Run:
- `python -m scripts.run_eval`

Note for corporate SSL/proxy networks:
- Evaluation defaults to local-only reranker loading (`RERANKER_LOCAL_FILES_ONLY=true`) so it does not try to download from Hugging Face.
- If you have trusted outbound internet and want cross-encoder downloads, set `RERANKER_LOCAL_FILES_ONLY=false` in `.env`.

Outputs:
- `docs/ragas-scorecard.json`
- `docs/ragas-scorecard.md`

Pass criteria:
- Faithfulness >= 0.90
- Context Recall >= 0.85

## Gemini Notes (optional)
Set environment variables in `.env` to enable Gemini generation:
- `USE_GEMINI=true`
- `GEMINI_API_KEY=...`
- `GEMINI_MODEL_NAME=gemini-1.5-pro`

The `/estimate` endpoint provides rough cost/latency projection under configured rate limits.

## Escalation Paths
Escalate for human review when:
- confidence is low
- query is jurisdiction-specific
- query crosses policy boundary or legal interpretation
- citations are missing

All answers are recommendations and require human approval.

## LoRA Fine-Tuning for Reranker (Confidence Improvement)
You can fine-tune the reranker with LoRA to improve ranking quality, which can increase answer confidence.

1. Train LoRA adapter and merged model:
   - `python -m scripts.train_lora_reranker --merge-adapter`
   - Offline/local cache mode: `python -m scripts.train_lora_reranker --base-model <local_model_path> --local-files-only --merge-adapter`
2. Point runtime reranker to the merged model in `.env`:
   - `RERANKER_MODEL_NAME=models/reranker_lora_merged`
   - `RERANKER_LOCAL_FILES_ONLY=true`
3. Re-run evaluation:
   - `python -m scripts.run_eval`

Notes:
- Current training data is built from `data/golden_set.json` positives plus sampled negatives from `data/synthetic_corpus.jsonl`.
- You can tune `--epochs`, `--learning-rate`, and `--negatives-per-positive` for better results.

### One-Command Automation (Train -> .env Update -> Eval)
Run everything in one command:
- `python -m scripts.train_update_eval`

For corporate SSL/proxy or offline cache mode:
- `python -m scripts.train_update_eval --base-model <local_model_path> --local-files-only`

Useful options:
- `--skip-train` (only update `.env` + run eval)
- `--skip-eval` (train + update `.env`, no eval)
