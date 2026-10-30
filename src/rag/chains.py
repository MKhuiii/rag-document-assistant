from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.core.llm import get_llm
from src.rag.prompts import qa_prompt
from src.rag.rerankers import rerank_documents
from src.rag.retrievers import get_hybrid_retriever


def format_docs(docs: list[Document]) -> str:
    """Gộp nội dung các chunks tìm được thành 1 chuỗi văn bản."""
    if not docs:
        return "Không có tài liệu liên quan."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(all_documents: list[Document] | None = None):
    """Khởi tạo RAG Chain với Hybrid Search & Re-ranking."""
    retriever = get_hybrid_retriever(all_documents)
    llm = get_llm()

    def get_context(query: str) -> str:
        """Hàm helper: Tìm kiếm -> Re-rank -> Format văn bản."""
        docs = retriever.invoke(query)
        ranked_docs = rerank_documents(query=query, docs=docs, top_n=5)
        return format_docs(ranked_docs)

    # Định nghĩa LCEL Chain
    chain = (
        {
            "context": itemgetter("question") | RunnableLambda(get_context),
            "question": itemgetter("question"),
            "chat_history": lambda x: x.get("chat_history", ""),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain
