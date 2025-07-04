from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantConfig(BaseModel):
    host: str
    port: int
    collection_name: str


class OpenAIConfig(BaseModel):
    api_key: str
    qa_model: str
    embed_model: str


class LLAMAConfig(BaseModel):
    api_key: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )
    llama: LLAMAConfig
    openai: OpenAIConfig
    qdrant: QdrantConfig

    debug: bool = False

    host: str
    port: int = 8000


settings = Settings()  # type: ignore[call-arg]
