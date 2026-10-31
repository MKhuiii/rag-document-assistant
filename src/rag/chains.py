import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.core.llm import get_llm
from src.rag.prompts import FALLBACK, condense_prompt, expand_prompt, qa_prompt
from src.rag.rerankers import rerank_documents
from src.rag.retrievers import get_hybrid_retriever

# Điểm của reranker đa ngữ thường bão hòa (~0.99), nên ngưỡng 0.0 gần như không lọc gì.
# Việc quyết định "không có thông tin" do LLM trả NO_INFO đảm nhiệm.
RERANK_THRESHOLD = 0.0
RERANK_TOP_N = 5
MAX_EXPANDED_QUERIES = 3

_VI_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_ANSWER_RE = re.compile(r"ANSWER:\s*(.*)", re.IGNORECASE | re.DOTALL)


def detect_lang(text: str) -> str:
    """Đoán ngôn ngữ của câu hỏi."""
    return "Vietnamese" if _VI_CHARS.search(text) else "English"


def format_docs(docs: list[Document]) -> str:
    """Gộp nội dung các chunks tìm được thành 1 chuỗi văn bản."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def parse_queries(raw: str) -> list[str]:
    """Tách output của LLM thành danh sách truy vấn, bỏ số thứ tự/gạch đầu dòng."""
    queries = []
    for line in raw.splitlines():
        line = _LIST_PREFIX.sub("", line).strip()
        if line:
            queries.append(line)
    return queries[:MAX_EXPANDED_QUERIES]


def parse_answer(raw: str, lang: str) -> str:
    """Lấy phần ANSWER; nếu model báo NO_INFO thì trả fallback bằng code."""
    match = _ANSWER_RE.search(raw)
    if match:
        return match.group(1).strip()
    if "NO_INFO" in raw:
        return FALLBACK[lang]
    return raw.strip()  # model không theo định dạng: trả nguyên văn


def create_rag_chain(all_documents: list[Document] | None = None):
    """RAG Chain: condense -> expand -> hybrid retrieve -> rerank -> trả lời."""
    retriever = get_hybrid_retriever(all_documents)
    llm = get_llm()

    condense_chain = condense_prompt | llm | StrOutputParser()
    expand_chain = expand_prompt | llm | StrOutputParser()
    answer_chain = qa_prompt | llm | StrOutputParser()

    def run(inputs: dict) -> str:
        question = inputs["question"]
        history = inputs.get("user_questions", "")  # chỉ câu hỏi cũ của user
        lang = detect_lang(question)  # ngôn ngữ của câu hỏi GỐC

        # 1) Viết lại thành câu hỏi độc lập (chỉ khi có lịch sử)
        standalone = (
            condense_chain.invoke({"history": history, "question": question}).strip()
            if history
            else question
        )

        # 2) Sinh thêm truy vấn để tìm đủ các đoạn liên quan
        queries = [standalone] + parse_queries(
            expand_chain.invoke({"question": standalone})
        )

        # 3) Tìm kiếm bằng mọi truy vấn, gộp và loại trùng
        seen: set[str] = set()
        docs: list[Document] = []
        for q in queries:
            for d in retriever.invoke(q):
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    docs.append(d)

        # 4) Rerank theo câu hỏi độc lập, rồi lọc theo ngưỡng
        ranked = rerank_documents(query=standalone, docs=docs, top_n=RERANK_TOP_N)
        ranked = [
            d for d in ranked if d.metadata.get("rerank_score", 0.0) >= RERANK_THRESHOLD
        ]
        if not ranked:
            return FALLBACK[lang]

        # 5) Trả lời (prompt không thấy chat history)
        raw = answer_chain.invoke(
            {
                "context": format_docs(ranked),
                "question": question,
                "answer_language": lang,
            }
        )
        return parse_answer(raw, lang)

    return RunnableLambda(run)
