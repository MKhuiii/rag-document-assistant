
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


def rerank_documents(
    query: str, docs: list[Document], top_n: int = 5
) -> list[Document]:
    """Tái xếp hạng danh sách tài liệu sử dụng FlashRank Cross-Encoder."""
    if not docs:
        return []

    # Chuyển đổi dữ liệu sang định dạng yêu cầu của FlashRank
    pass_passages = [
        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
        for idx, doc in enumerate(docs)
    ]

    rerank_req = RerankRequest(query=query, passages=pass_passages)
    results = ranker.rerank(rerank_req)

    # Lọc lấy Top-N tài liệu có điểm cao nhất
    ranked_docs = []
    for item in results[:top_n]:
        doc = Document(
            page_content=item["text"],
            metadata={**item["meta"], "rerank_score": item["score"]},
        )
        ranked_docs.append(doc)

    return ranked_docs
