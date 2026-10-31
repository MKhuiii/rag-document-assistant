from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BACKEND_URL: str = "http://127.0.0.1:8000"

    # Database
    POSTGRES_USER: str = "rag"
    POSTGRES_PASSWORD: str = "rag"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "rag"
    DATABASE_URL: str = ""

    # Ollama
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_HOST: str = "http://127.0.0.1:11434"

    # Embedding
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIM: int = 1024

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def db_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings()
