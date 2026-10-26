from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.core.vector_db import get_vector_store


def get_hybrid_retriever(documents: list[Document] | None = None) -> BaseRetriever:
    """Khởi tạo Hybrid Retriever kết hợp BM25 (Sparse) và PGVector (Dense)."""
    # 1. Dense Retriever từ PGVector
    vectorstore = get_vector_store()
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # Nếu không có tài liệu truyền vào để build BM25, fallback về Vector Retriever thuần
    if not documents:
        return dense_retriever

    # 2. Sparse Retriever từ BM25
    sparse_retriever = BM25Retriever.from_documents(documents)
    sparse_retriever.k = 15

    # 3. Kết hợp bằng EnsembleRetriever (RRF)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4],  # Ưu tiên Vector Search 60%, BM25 40%
    )
    return ensemble_retriever
