from langchain_core.documents import Document

from src.rag.rerankers import rerank_documents


def test_flashrank_reranker():
    query = "Quy định về điểm rèn luyện sinh viên"
    docs = [
        Document(page_content="Quy định về học phí và học bổng năm 2026."),
        Document(
            page_content="Hướng dẫn đánh giá điểm rèn luyện cho sinh viên các khóa."
        ),
        Document(page_content="Lịch thi học kỳ và đăng ký môn học."),
    ]

    reranked = rerank_documents(query=query, docs=docs, top_n=2)

    assert len(reranked) == 2
    # Document về điểm rèn luyện phải được xếp lên đầu tiên (Top 1)
    assert "điểm rèn luyện" in reranked[0].page_content
    assert "rerank_score" in reranked[0].metadata
