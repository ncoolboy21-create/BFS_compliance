from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Bank Compliance RAG"
    app_version: str = "0.1.0"

    data_path: str = "data/synthetic_corpus.jsonl"
    top_k_retrieval: int = 12
    top_k_rerank: int = 5
    chunk_size_chars: int = 700
    chunk_overlap_chars: int = 120
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_name: str = "compliance_chunks"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_local_files_only: bool = False

    use_gemini: bool = Field(default=False, description="Set true to call Gemini.")
    gemini_api_key: str | None = None
    gemini_model_name: str = "gemini-1.5-pro"
    gemini_temperature: float = 0.2

    gemini_rpm_limit: int = 60
    gemini_tpm_limit: int = 240000
    gemini_input_cost_per_million: float = 1.25
    gemini_output_cost_per_million: float = 5.00

    use_azure_openai: bool = Field(default=True, description="Set true to call Azure OpenAI.")
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_temperature: float = 0.4

    use_openai: bool = Field(default=False, description="Set true to call OpenAI API directly.")
    openai_api_key: str | None = None
    openai_model_name: str = "gpt-4o-mini"
    openai_temperature: float = 0.2

    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_local_files_only: bool = True

    low_confidence_threshold: float = 0.65
    auto_approval_confidence_threshold: float = 0.60


settings = Settings()
