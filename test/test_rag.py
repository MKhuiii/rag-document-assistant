from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag.chains import create_rag_chain
from src.rag.prompts import FALLBACK


def _make_chain(llm_text: str, retrieved: list[Document], ranked: list[Document]):
    """Dựng chain với retriever, LLM và reranker đều là mock."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = retrieved

    mock_llm = MagicMock(return_value=AIMessage(content=llm_text))

    patches = (
        patch("src.rag.chains.get_hybrid_retriever", return_value=mock_retriever),
        patch("src.rag.chains.get_llm", return_value=mock_llm),
        patch("src.rag.chains.rerank_documents", return_value=ranked),
    )
    return patches


def _run(llm_text, retrieved, ranked, question="Dự án này sử dụng mô hình gì?"):
    p1, p2, p3 = _make_chain(llm_text, retrieved, ranked)
    with p1, p2, p3:
        chain = create_rag_chain()
        return chain.invoke({"question": question})


def test_rag_chain_returns_answer_line():
    """Có tài liệu liên quan: trả về phần ANSWER do LLM sinh ra."""
    doc = Document(
        page_content="Nội dung tài liệu giả lập.",
        metadata={"rerank_score": 0.9},
    )
    result = _run(
        "FACTS: Nội dung giả lập.\nANSWER: Đây là câu trả lời giả lập từ LLM.",
        retrieved=[doc],
        ranked=[doc],
    )
    assert result == "Đây là câu trả lời giả lập từ LLM."


def test_rag_chain_no_docs_returns_fallback():
    """Không tìm được tài liệu: trả fallback bằng code, không cần gọi LLM trả lời."""
    result = _run("bất kỳ", retrieved=[], ranked=[])
    assert result == FALLBACK["Vietnamese"]


def test_rag_chain_no_info_returns_fallback():
    """LLM báo NO_INFO: code đổi thành câu fallback đúng ngôn ngữ."""
    doc = Document(page_content="Không liên quan.", metadata={"rerank_score": 0.9})
    result = _run("NO_INFO", retrieved=[doc], ranked=[doc])
    assert result == FALLBACK["Vietnamese"]


def test_rag_chain_english_fallback():
    """Câu hỏi tiếng Anh: fallback phải bằng tiếng Anh."""
    result = _run(
        "bất kỳ", retrieved=[], ranked=[], question="Who is the head of the department?"
    )
    assert result == FALLBACK["English"]


if __name__ == "__main__":
    print("--- ĐANG CHẠY BÀI TEST TRUY VẤN RAG THỰC TẾ (LOCAL) ---")
    rag_chain = create_rag_chain()

    query = "Dự án Local RAG Assistant sử dụng các công nghệ chính nào?"
    print(f"\nCâu hỏi: {query}")
    print("Đang suy luận...\n")

    response = rag_chain.invoke({"question": query, "user_questions": ""})
    print("Trả lời:")
    print(response)
