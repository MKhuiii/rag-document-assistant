import requests
import streamlit as st

from src.config import settings

API_URL = f"{settings.BACKEND_URL}/api/v1"

st.set_page_config(page_title="Local RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Trợ Lý Tra Cứu Tài Liệu Cá Nhân (Local RAG)")

# --- SIDEBAR: Nạp tài liệu ---
with st.sidebar:
    st.header("📄 Quản lý Tài liệu")
    uploaded_file = st.file_uploader(
        "Tải lên tài liệu (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"]
    )

    if uploaded_file is not None and st.button(
        "Tiến hành Nạp Dữ Liệu", use_container_width=True
    ):
        with st.spinner("Đang phân tách và tạo Vector Embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            res = requests.post(f"{API_URL}/uploads", files=files)
            if res.status_code == 200:
                st.success(res.json().get("message"))
            else:
                st.error(f"Lỗi: {res.json().get('detail')}")

# --- MAIN: Khung Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lại lịch sử chat trên giao diện
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
if prompt := st.chat_input("Hỏi bất kỳ điều gì về tài liệu của bạn..."):
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Gọi API FastAPI
    with (
        st.chat_message("assistant"),
        st.spinner("Đang tra cứu tài liệu & suy luận..."),
    ):
        history_data = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        try:
            res = requests.post(
                f"{API_URL}/chats",
                json={"question": prompt, "history": history_data},
                timeout=120,  # Cho phép tối đa 120s để LLM suy luận
            )
            if res.status_code == 200:
                answer = res.json()["answer"]
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                # Hiển thị chính xác thông báo lỗi từ FastAPI trả về
                detail = res.json().get("detail", res.text)
                st.error(f"Lỗi Server ({res.status_code}): {detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi kết nối tới Server API: {e}")
