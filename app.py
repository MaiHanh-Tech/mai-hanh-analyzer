import streamlit as st
import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import numpy as np
import os
import time
from datetime import datetime
from collections import defaultdict

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Mai Hanh Super App", layout="wide", page_icon="💎")

# --- 2. CLASS QUẢN LÝ MẬT KHẨU (TÍCH HỢP SẴN) ---
class PasswordManager:
    def __init__(self):
        # Lấy thông tin từ secrets
        self.user_tiers = st.secrets.get("user_tiers", {})
        
        # Khởi tạo session state nếu chưa có
        if 'usage_tracking' not in st.session_state:
            st.session_state.usage_tracking = {}
        if 'key_name_mapping' not in st.session_state:
            st.session_state.key_name_mapping = {}
            
    def check_password(self, password):
        """Kiểm tra mật khẩu nhập vào"""
        if not password: return False
        
        # 1. Kiểm tra Admin
        admin_pwd = st.secrets.get("admin_password")
        if password == admin_pwd:
            st.session_state.key_name_mapping[password] = "admin"
            return True
        
        # 2. Kiểm tra User thường (Từ danh sách api_keys)
        api_keys = st.secrets.get("api_keys", {})
        for key_name, key_value in api_keys.items():
            if password == key_value:
                st.session_state.key_name_mapping[password] = key_name
                return True
        return False
        
    def is_admin(self, password):
        return password == st.secrets.get("admin_password")

# --- 3. CÁC HÀM XỬ LÝ AI & FILE (CORE) ---
@st.cache_resource
def load_models():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def doc_file(uploaded_file):
    if not uploaded_file: return ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext == '.pdf':
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif ext == '.docx':
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in ['.txt', '.md']:
            return str(uploaded_file.read(), "utf-8")
        elif ext in ['.html', '.htm']:
            soup = BeautifulSoup(uploaded_file, 'html.parser')
            return soup.get_text()
    except: return ""
    return ""

def luu_lich_su(loai, tieu_de, noi_dung):
    thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "time": thoi_gian, "type": loai, "title": tieu_de, "content": noi_dung
    })

# --- 4. GIAO DIỆN SIÊU ỨNG DỤNG (SAU KHI LOGIN) ---
def show_main_app():
    # Khởi tạo bộ nhớ
    if 'history' not in st.session_state: st.session_state.history = []
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []

    # Cấu hình API Gemini (Lấy từ Secrets chung)
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        model = genai.GenerativeModel('gemini-2.5-pro') # Dùng bản Flash cho nhanh
    except:
        st.error("❌ Lỗi: Chưa cấu hình Gemini API Key trong Secrets!")
        st.stop()

    # --- SIDEBAR: LOGOUT & INFO ---
    with st.sidebar:
        st.success(f"👤 Chào mừng: {st.session_state.current_user_name}")
        if st.button("Đăng Xuất (Logout)"):
            st.session_state.user_logged_in = False
            st.session_state.current_user = None
            st.rerun()
    
    st.title("💎 The Mai Hanh Super-App")

    # --- TABS CHỨC NĂNG ---
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Phân Tích Sách", "✍️ Dịch Giả Xịn", "🗣️ Tranh Biện", "⏳ Lịch Sử"])

    # TAB 1: PHÂN TÍCH SÁCH
    with tab1:
        st.header("Trợ lý Nghiên cứu RAG")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            file_excel = st.file_uploader("1. Kết nối Kho Sách (Excel)", type="xlsx", key="tab1_excel")
            uploaded_files = st.file_uploader("2. Tài liệu mới", type=["pdf","docx","txt"], accept_multiple_files=True)
            if st.button("🚀 Phân Tích"):
                if uploaded_files:
                    # Logic Vector (Rút gọn)
                    vec_model = load_models()
                    db_vec, df = None, None
                    if file_excel:
                        df = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
                        content = [f"{r['Tên sách']} {r['CẢM NHẬN']}" for i,r in df.iterrows()]
                        db_vec = vec_model.encode(content)
                    
                    for f in uploaded_files:
                        text = doc_file(f)
                        lien_ket = ""
                        if db_vec is not None:
                            q_vec = vec_model.encode([text[:1000]])
                            scores = cosine_similarity(q_vec, db_vec)[0]
                            top = np.argsort(scores)[::-1][:3]
                            for idx in top:
                                if scores[idx] > 0.35: lien_ket += f"- {df.iloc[idx]['Tên sách']}\n"
                        
                        prompt = f"Phân tích tài liệu '{f.name}'. Liên kết cũ: {lien_ket}. Nội dung: {text[:20000]}"
                        res = model.generate_content(prompt)
                        st.markdown(f"### {f.name}\n{res.text}")
                        luu_lich_su("Phân Tích", f.name, res.text)

    # TAB 2: DỊCH GIẢ (TỰ ĐỘNG)
    with tab2:
        st.header("Dịch Thuật Đa Chiều")
        c1, c2 = st.columns(2)
        with c1:
            txt_in = st.text_area("Nhập văn bản (Việt/Anh/Trung):", height=200)
            if st.button("Dịch Ngay"):
                with st.spinner("Đang xử lý..."):
                    prompt = f"""
                    Bạn là Chuyên gia Ngôn ngữ. Xử lý văn bản: "{txt_in}"
                    Logic:
                    - Nếu là Tiếng Việt -> Dịch sang Anh & Trung (kèm Pinyin).
                    - Nếu là Ngoại ngữ -> Dịch sang Tiếng Việt (Văn phong hay).
                    - Phân tích 3 từ vựng hay nhất.
                    """
                    res = model.generate_content(prompt)
                    with c2: st.markdown(res.text)
                    luu_lich_su("Dịch Thuật", txt_in[:20], res.text)

    # TAB 3: TRANH BIỆN
    with tab3:
        st.header("Luyện Tư Duy Phản Biện")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])
        
        if query := st.chat_input("Chủ đề tranh luận..."):
            st.chat_message("user").markdown(query)
            st.session_state.chat_history.append({"role":"user", "content":query})
            
            prompt = f"Phản biện lại quan điểm này một cách sâu sắc: '{query}'"
            res = model.generate_content(prompt)
            
            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append({"role":"assistant", "content":res.text})

    # TAB 4: LỊCH SỬ
    with tab4:
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                with st.expander(f"⏰ {item['time']} | {item['type']} | {item['title']}"):
                    st.markdown(item['content'])
        else:
            st.info("Chưa có lịch sử.")

# --- 5. HÀM MAIN (ĐIỀU PHỐI LOGIN) ---
def main():
    # Khởi tạo Password Manager
    pm = PasswordManager()

    # Kiểm tra trạng thái đăng nhập
    if not st.session_state.get('user_logged_in', False):
        # --- MÀN HÌNH ĐĂNG NHẬP ---
        st.title("🔐 Mai Hạnh Super-App Login")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            user_pass = st.text_input("Nhập Mật Khẩu Truy Cập:", type="password")
            if st.button("Đăng Nhập", type="primary", use_container_width=True):
                if pm.check_password(user_pass):
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = user_pass
                    st.session_state.current_user_name = st.session_state.key_name_mapping.get(user_pass, "User")
                    st.session_state.is_admin = pm.is_admin(user_pass)
                    st.rerun()
                else:
                    st.error("Sai mật khẩu rồi Sếp ơi!")
    else:
        # --- ĐÃ ĐĂNG NHẬP -> VÀO APP ---
        show_main_app()

if __name__ == "__main__":
    main()
