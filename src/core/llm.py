from langchain_ollama import ChatOllama

from src.config import settings


def get_llm() -> ChatOllama:
    """
    Khởi tạo mô hình ngôn ngữ từ Ollama.
    - temperature=0: Giúp AI trả lời chính xác, bám sát tài liệu thay vì bịa.
    """
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_HOST,
        temperature=0,
        num_ctx=8192,
    )
