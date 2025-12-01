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
import gspread # Thư viện Google Sheets
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Mai Hanh Super App", layout="wide", page_icon="💎")

# --- 2. CLASS QUẢN LÝ MẬT KHẨU ---
class PasswordManager:
    def __init__(self):
        self.user_tiers = st.secrets.get("user_tiers", {})
        if 'key_name_mapping' not in st.session_state:
            st.session_state.key_name_mapping = {}
            
    def check_password(self, password):
        if not password: return False
        admin_pwd = st.secrets.get("admin_password")
        if password == admin_pwd:
            st.session_state.key_name_mapping[password] = "admin"
            return True
        api_keys = st.secrets.get("api_keys", {})
        for key_name, key_value in api_keys.items():
            if password == key_value:
                st.session_state.key_name_mapping[password] = key_name
                return True
        return False
    
    def is_admin(self, password):
        return password == st.secrets.get("admin_password")

# --- 3. DATABASE MANAGER (GOOGLE SHEETS) ---
# Hàm này giúp kết nối với "Ổ cứng"
# --- SỬA LẠI HÀM NÀY ĐỂ BẮT LỖI ---
def connect_gsheet():
    try:
        # Lấy thông tin Service Account từ Secrets
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Kiểm tra xem có secrets chưa
        if "gcp_service_account" not in st.secrets:
            st.error("❌ LỖI: Chưa cấu hình [gcp_service_account] trong Secrets!")
            return None

        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Mở file sheet theo tên
        sheet = client.open("AI_History_Logs").sheet1 
        return sheet
        
    except Exception as e:
        # IN LỖI RA MÀN HÌNH ĐỂ CHỊ THẤY
        st.error(f"❌ KHÔNG KẾT NỐI ĐƯỢC GOOGLE SHEET: {e}")
        return None

def luu_lich_su_vinh_vien(loai, tieu_de, noi_dung):
    thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Lưu vào RAM (để hiện ngay lập tức)
    if 'history' not in st.session_state: st.session_state.history = []
    st.session_state.history.append({"time": thoi_gian, "type": loai, "title": tieu_de, "content": noi_dung})
    
    # 2. Lưu vào Google Sheets (Ổ cứng)
    try:
        sheet = connect_gsheet()
        if sheet:
            sheet.append_row([thoi_gian, loai, tieu_de, noi_dung])
    except Exception as e:
        print(f"Lỗi lưu Sheet: {e}") # Chỉ in lỗi ngầm, không làm phiền user

def tai_lich_su_tu_sheet():
    try:
        sheet = connect_gsheet()
        if sheet:
            data = sheet.get_all_records()
            # Chuyển đổi key về chữ thường để khớp logic cũ
            formatted_data = []
            for item in data:
                formatted_data.append({
                    "time": item.get("Time", ""),
                    "type": item.get("Type", ""),
                    "title": item.get("Title", ""),
                    "content": item.get("Content", "")
                })
            return formatted_data
    except:
        return []
    return []

# --- 4. CÁC HÀM XỬ LÝ AI ---
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

# --- 5. GIAO DIỆN CHÍNH ---
def show_main_app():
    # Tải lịch sử từ Cloud về khi mở App
    if 'history_loaded' not in st.session_state:
        cloud_history = tai_lich_su_tu_sheet()
        if cloud_history:
            st.session_state.history = cloud_history
        elif 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history_loaded = True

    if 'chat_history' not in st.session_state: st.session_state.chat_history = []

    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
    except:
        st.error("❌ Lỗi: Chưa cấu hình Gemini API Key!")
        st.stop()

    with st.sidebar:
        st.success(f"👤 User: {st.session_state.current_user_name}")
        if st.button("Logout"):
            st.session_state.user_logged_in = False
            st.session_state.current_user = None
            st.rerun()
    
    st.title("💎 The Mai Hanh Super-App")
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Phân Tích Sách", "✍️ Dịch Giả", "🗣️ Tranh Biện", "⏳ Lịch Sử "])

    # TAB 1: PHÂN TÍCH
    with tab1:
        st.header("Trợ lý Nghiên cứu RAG")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            file_excel = st.file_uploader("1. Kết nối Kho Sách", type="xlsx", key="tab1_excel")
            uploaded_files = st.file_uploader("2. Tài liệu mới", type=["pdf","docx","txt"], accept_multiple_files=True)
            if st.button("🚀 Phân Tích"):
                if uploaded_files:
                    vec_model = load_models()
                    db_vec, df = None, None
                    if file_excel:
                        try:
                            df = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
                            content = [f"{r['Tên sách']} {r['CẢM NHẬN']}" for i,r in df.iterrows()]
                            db_vec = vec_model.encode(content)
                        except: pass
                    
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
                        # LƯU VĨNH VIỄN
                        luu_lich_su_vinh_vien("Phân Tích", f.name, res.text)

    # TAB 2: DỊCH GIẢ
    with tab2:
        st.header("Dịch Thuật Đa Chiều")
        c1, c2 = st.columns(2)
        with c1:
            txt_in = st.text_area("Nhập văn bản:", height=200)
            if st.button("Dịch Ngay"):
                with st.spinner("Đang xử lý..."):
                    prompt = f"Dịch và phân tích (Việt/Anh/Trung) cho văn bản: '{txt_in}'"
                    res = model.generate_content(prompt)
                    with c2: st.markdown(res.text)
                    # LƯU VĨNH VIỄN
                    luu_lich_su_vinh_vien("Dịch Thuật", txt_in[:20], res.text)

    # TAB 3: TRANH BIỆN
    with tab3:
        st.header("Luyện Tư Duy")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])
        
        if query := st.chat_input("Chủ đề tranh luận..."):
            st.chat_message("user").markdown(query)
            st.session_state.chat_history.append({"role":"user", "content":query})
            
            prompt = f"Phản biện: '{query}'"
            res = model.generate_content(prompt)
            
            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append({"role":"assistant", "content":res.text})
            # Chat thì lưu vào DB hơi tốn, nên chỉ lưu vào RAM hoặc lưu cuối phiên

    # TAB 4: LỊCH SỬ (ĐỌC TỪ CLOUD)
    with tab4:
        st.header("Kho Lưu Trữ (Google Sheets)")
        if st.button("🔄 Tải lại Lịch sử từ Cloud"):
            st.session_state.history = tai_lich_su_tu_sheet()
            st.rerun()
            
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                with st.expander(f"⏰ {item['time']} | {item['type']} | {item['title']}"):
                    st.markdown(item['content'])
        else:
            st.info("Chưa có lịch sử hoặc chưa kết nối Database.")

# --- 6. MAIN ---
def main():
    pm = PasswordManager()
    if not st.session_state.get('user_logged_in', False):
        st.title("🔐 Login")
        user_pass = st.text_input("Password:", type="password")
        if st.button("Login"):
            if pm.check_password(user_pass):
                st.session_state.user_logged_in = True
                st.session_state.current_user = user_pass
                st.session_state.current_user_name = st.session_state.key_name_mapping.get(user_pass, "User")
                st.rerun()
            else: st.error("Sai mật khẩu!")
    else:
        show_main_app()

if __name__ == "__main__":
    main()
