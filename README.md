# 🤖 Local RAG Document Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?logo=ollama)
![License](https://img.shields.io/badge/License-MIT-green.svg)


## 🌟 Highlights

- **100% Local & Privacy-First:** Chạy hoàn toàn Offline trên máy cá nhân với Ollama (`qwen2.5:7b`, `bge-m3`) và PostgreSQL (`pgvector`), đảm bảo tuyệt đối an toàn dữ liệu.

- **Multi-Format Document Ingestion:** Tự động đọc, phân tách (chunking) và đánh chỉ mục vector cho các định dạng tài liệu phổ biến (`.pdf`, `.txt`, `.docx`).

- **Production-Ready REST API:** Xây dựng trên FastAPI chuẩn hóa kiến trúc Route, hỗ trợ Dependency Injection, ThreadPool cách ly I/O blocking và xử lý lịch sử trò chuyện (Chat History).

- **Intuitive Web Interface:** Giao diện Streamlit mượt mà, hỗ trợ tải tài liệu và trò chuyện tương tác trực tiếp với RAG Chain.

- **High Code Quality & Test Suite:** Đạt chuẩn kiểm tra mã nguồn khắt khe với Ruff Linter và tích hợp bộ kiểm thử tự động Pytest (Mock & Real Integration).


## ℹ️ Overview

**Local RAG Document Assistant** là một trợ lý ảo tra cứu tài liệu cá nhân được xây dựng dựa trên kiến trúc Retrieval-Augmentation Generation (RAG). Dự án giúp bạn hỏi đáp và trích xuất thông tin trực tiếp từ các văn bản nội bộ một cách nhanh chóng, chính xác mà không lo bị rò rỉ dữ liệu ra các dịch vụ AI Cloud bên ngoài. Hệ thống áp dụng LangChain Expression Language (LCEL) để tối ưu luồng xử lý và đảm bảo hiệu năng tối đa.


### ✍️ Authors

- **Maintainer:** MKhuiii (`@MKhuiii`)
- **GitHub Repository:** [RAG Document Assistant](https://github.com/MKhuiii/rag-document-assistant)




## 🚀 Usage

Khởi chạy hệ thống RAG Assistant chỉ với vài câu lệnh đơn giản:

```bash
# 1. Khởi chạy PostgreSQL với pgvector
docker compose up -d

# 2. Khởi chạy FastAPI Backend Server (Port 8000)
uvicorn src.main:app --reload --port 8000

# 3. Khởi chạy Streamlit Web UI (Port 8501)
streamlit run app.py

```
## ⬇️ Installation

**Tải và cài đặt Ollama:**
   - Tải bản cài đặt phù hợp với hệ điều hành (Windows / macOS / Linux) tại [ollama.com/download](https://ollama.com/download).

**Khởi chạy Dịch vụ Ollama:**
   - Đảm bảo ứng dụng Ollama đang chạy hoặc khởi chạy qua lệnh:
     ```bash
     ollama serve
     ```

**Tải các mô hình AI phục vụ dự án:**
   Chạy các lệnh sau trong Terminal / PowerShell để tải 2 mô hình bắt buộc:
   ```bash
   # Mô hình Embedding (Tạo vector cho văn bản)
   ollama pull bge-m3

   # Mô hình LLM (Suy luận & Trả lời câu hỏi)
   ollama pull qwen2.5:7b
   ```