from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    database_url: str = "postgresql+asyncpg://ragcore:ragcore@localhost:5432/ragcore"
    redis_url: str = "redis://localhost:6379/0"

    chroma_host: str = "localhost"
    chroma_port: int = 8000

    allowed_origins: list[str] = ["http://localhost:3000"]
    api_key_header: str = "X-API-Key"

    MAX_UPLOAD_SIZE_MB: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
