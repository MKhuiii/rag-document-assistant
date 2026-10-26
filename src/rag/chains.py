from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.core.llm import get_llm
from src.rag.prompts import qa_prompt
from src.rag.retrievers import get_hybrid_retriever


def format_docs(docs: list[Document]) -> str:
    """Gộp nội dung các chunks tìm được thành 1 chuỗi văn bản."""
    if not docs:
        return "Không có tài liệu liên quan."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    """Khởi tạo RAG Chain bằng LCEL hỗ trợ Chat History."""
    retriever = get_hybrid_retriever()
    llm = get_llm()

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain
