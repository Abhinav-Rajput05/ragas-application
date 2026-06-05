from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False,
    )

    nexus_api_key: str
    nexus_base_url: str = "https://apidev.navigatelabsai.com"
    nexus_model: str = "gpt-4.1-nano"

    chroma_persist_dir: str = "./data/chroma"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    max_optimization_configs: int = 12

    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    default_top_k: int = 5
    default_embedding_model: str = "all-MiniLM-L6-v2"

    opt_chunk_sizes: list[int] = [256, 512, 1024]
    opt_chunk_overlaps: list[int] = [20, 50, 100]
    opt_top_k_values: list[int] = [3, 5, 10]
    opt_embedding_models: list[str] = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "BAAI/bge-m3",
    ]

    weight_faithfulness: float = 0.30
    weight_answer_relevancy: float = 0.25
    weight_context_recall: float = 0.25
    weight_context_precision: float = 0.20

    threshold_faithfulness: float = 0.75
    threshold_hallucination_rate: float = 0.20
    threshold_context_recall: float = 0.70
    threshold_health_score_ready: float = 80.0
    threshold_health_score_needs_work: float = 60.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
