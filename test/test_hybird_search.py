from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


def rerank_documents(
    query: str, docs: list[Document], top_n: int = 5
) -> list[Document]:
    """Tái xếp hạng bằng bge-reranker-v2-m3. Điểm (0-1) lưu trong metadata["rerank_score"]."""
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = _get_model().predict(pairs)

    scored = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": float(score)},
        )
        for doc, score in scored[:top_n]
    ]
