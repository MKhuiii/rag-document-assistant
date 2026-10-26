from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.rag.chains import create_rag_chain


def test_rag_chain_mock():
    """Test cấu trúc RAG Chain với Mock trong môi trường CI."""

    mock_retriever = MagicMock()
    mock_retriever.return_value = []
    mock_retriever.invoke.return_value = []

    mock_response = AIMessage(content="Đây là câu trả lời giả lập từ LLM.")
    mock_llm = MagicMock()
    mock_llm.return_value = mock_response
    mock_llm.invoke.return_value = mock_response

    with (
        patch("src.rag.chains.get_hybrid_retriever", return_value=mock_retriever),
        patch("src.rag.chains.get_llm", return_value=mock_llm),
    ):
        chain = create_rag_chain()
        result = chain.invoke(
            {
                "question": "Dự án này sử dụng mô hình gì?",
                "chat_history": "Chưa có lịch sử.",
            }
        )
        assert isinstance(result, str)
        assert result == "Đây là câu trả lời giả lập từ LLM."


if __name__ == "__main__":
    print("--- ĐANG CHẠY BÀI TEST TRUY VẤN RAG THỰC TẾ (LOCAL) ---")
    rag_chain = create_rag_chain()

    query = "Dự án Local RAG Assistant sử dụng các công nghệ chính nào?"
    print(f"\nCâu hỏi: {query}")
    print("Đang suy luận...\n")

    response = rag_chain.invoke({"question": query, "chat_history": "Chưa có lịch sử."})
    print("Trả lời:")
    print(response)
