from langchain_postgres import PGVector
from sqlalchemy import create_engine

from src.config import settings
from src.core.embeddings import get_embedding_model

engine = create_engine(
    settings.db_url,
    pool_pre_ping=True,  # Kiểm tra sống/chết của socket trước mỗi query
    pool_recycle=300,  # Tự động làm mới kết nối sau mỗi 5 phút
    pool_size=10,
    max_overflow=20,
)


def get_vector_store() -> PGVector:
    """
    Khởi tạo kết nối tới PostgreSQL sử dụng extension pgvector.
    LangChain tự động kết hợp mô hình embedding vào database để mã hóa dữ liệu đầu vào.
    """
    return PGVector(
        embeddings=get_embedding_model(),
        collection_name="rag_documents",
        connection=engine,
        use_jsonb=True,
    )
