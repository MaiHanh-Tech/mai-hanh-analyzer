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
import gspread 
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
        
        # Check Admin
        admin_pwd = st.secrets.get("admin_password")
        if password == admin_pwd:
            st.session_state.key_name_mapping[password] = "admin"
            return True
            
        # Check User
        api_keys = st.secrets.get("api_keys", {})
        for key_name, key_value in api_keys.items():
            if password == key_value:
                st.session_state.key_name_mapping[password] = key_name
                return True
        return False
    
    def is_admin(self, password):
        return password == st.secrets.get("admin_password")

# --- 3. DATABASE MANAGER (GOOGLE SHEETS - ĐÃ FIX LỖI KEY) ---
def connect_gsheet():
    try:
        if "gcp_service_account" not in st.secrets:
            return None

        # Lấy thông tin và FIX LỖI KHÓA (Quan trọng)
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            # Tự động sửa lỗi xuống dòng khi copy paste
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n").replace('\\n', '\n')

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Mở file
        return client.open("AI_History_Logs").sheet1 
    except Exception as e:
        return None

def luu_lich_su_vinh_vien(loai, tieu_de, noi_dung):
    thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Lưu RAM
    if 'history' not in st.session_state: st.session_state.history = []
    st.session_state.history.append({"time": thoi_gian, "type": loai, "title": tieu_de, "content": noi_dung})
    
    # 2. Lưu Cloud
    try:
        sheet = connect_gsheet()
        if sheet:
            sheet.append_row([thoi_gian, loai, tieu_de, noi_dung])
    except: pass 

def tai_lich_su_tu_sheet():
    try:
        sheet = connect_gsheet()
        if sheet:
            data = sheet.get_all_records()
            formatted = []
            for item in data:
                formatted.append({
                    "time": item.get("Time", ""),
                    "type": item.get("Type", ""),
                    "title": item.get("Title", ""),
                    "content": item.get("Content", "")
                })
            return formatted
    except: return []
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
    # Tải lịch sử
    if 'history_loaded' not in st.session_state:
        cloud_data = tai_lich_su_tu_sheet()
        if cloud_data: st.session_state.history = cloud_data
        st.session_state.history_loaded = True
    
    if 'history' not in st.session_state: st.session_state.history = []
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []

    # Cấu hình Gemini
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
    except:
        st.error("❌ Lỗi: Chưa cấu hình [system] gemini_api_key trong Secrets!")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.success(f"👤 User: {st.session_state.current_user_name}")
        
        # NÚT KIỂM TRA KẾT NỐI (DEBUG)
        with st.expander("🛠️ Công cụ Kỹ thuật"):
            if st.button("Test Kết nối Google Sheet"):
                sheet = connect_gsheet()
                if sheet:
                    st.success(f"✅ OK! Đã thấy file: {sheet.title}")
                    try:
                        sheet.append_row(["TEST", "System Check", "OK", str(datetime.now())])
                        st.info("Đã ghi thử 1 dòng.")
                    except: st.error("Kết nối được nhưng không ghi được (Quyền Editor?).")
                else:
                    st.error("❌ Kết nối thất bại. Kiểm tra lại Secrets/Email Robot.")

        if st.button("Đăng Xuất"):
            st.session_state.user_logged_in = False
            st.rerun()

    st.title("💎 The Mai Hanh Super-App")
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Phân Tích Sách", "✍️ Dịch Giả", "🗣️ Tranh Biện", "⏳ Lịch Sử"])

    # TAB 1: PHÂN TÍCH
    with tab1:
        st.header("Trợ lý Nghiên cứu RAG & Biểu đồ")
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            file_excel = st.file_uploader("1. Kết nối Kho Sách", type="xlsx", key="tab1_excel")
            uploaded_files = st.file_uploader("2. Tài liệu mới", type=["pdf","docx","txt", "md", "html"], accept_multiple_files=True)
            if st.button("🚀 PHÂN TÍCH (RAG)", type="primary"):
                if uploaded_files:
                    
                    # --- CHẠY PHÂN TÍCH TÀI LIỆU MỚI (RAG) ---
                    vec_model = load_models()
                    db_vec, df = None, None
                    has_db = False
                    
                    if file_excel:
                        try:
                            df = pd.read_excel(file_excel).dropna(subset=['Tên sách', 'CẢM NHẬN'])
                            if not df.empty:
                                st.info(f"✅ Đã tải {len(df)} sách. Đang số hóa...")
                                content = [f"{r['Tên sách']} {str(r['CẢM NHẬN'])}" for i,r in df.iterrows()]
                                db_vec = vec_model.encode(content)
                                has_db = True
                        except: st.error("❌ Lỗi đọc file Excel hoặc thiếu cột.")

                    # Lặp qua từng file mới
                    for f in uploaded_files:
                        text = doc_file(f)
                        lien_ket = ""
                        
                        # LOGIC RAG: TÌM KIẾM LIÊN KẾT TRONG KHO SÁCH CŨ
                        if has_db:
                            # Chỉ lấy 20.000 ký tự đầu để so sánh Vector cho nhanh
                            q_vec = vec_model.encode([text[:20000]]) 
                            scores = cosine_similarity(q_vec, db_vec)[0]
                            top = np.argsort(scores)[::-1][:3]
                            for idx in top:
                                if scores[idx] > 0.40: # Ngưỡng 40% để liên kết
                                    lien_ket += f"- {df.iloc[idx]['Tên sách']} (Khớp: {scores[idx]*100:.1f}%)\n"
                        
                        # CHẠY GEMINI PHÂN TÍCH
                        prompt = f"Phân tích tài liệu '{f.name}'. Nguồn liên kết cũ: {lien_ket}. Nội dung: {text[:20000]}"
                        res = model.generate_content(prompt)
                        st.markdown(f"### 📄 Kết quả: {f.name}\n{res.text}")
                        luu_lich_su_vinh_vien("Phân Tích", f.name, res.text)
        
        # --- [NEW] PHẦN HIỂN THỊ BIỂU ĐỒ (LUÔN CHẠY KHI CÓ EXCEL) ---
        if file_excel:
            import plotly.express as px
            # Phải đọc lại file (hoặc dùng biến df nếu nó đã được định nghĩa ở trên)
            # Dùng st.session_state để lưu df nếu không nó sẽ bị mất
            if 'df_kho_sach' not in st.session_state and 'df' in locals():
                st.session_state.df_kho_sach = df
            
            if 'df_kho_sach' in st.session_state:
                df_viz = st.session_state.df_kho_sach
                
                with st.expander("📊 Thống Kê & Bản Đồ Kho Sách", expanded=True):
                    # 1. Biểu đồ Top Tác giả
                    if 'Tác giả' in df_viz.columns:
                        top_authors = df_viz['Tác giả'].value_counts().head(10).reset_index()
                        top_authors.columns = ['Tác giả', 'Số lượng']
                        fig2 = px.bar(top_authors, x='Tác giả', y='Số lượng', title='Top 10 Tác giả Chị đọc nhiều nhất')
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # 2. Biểu đồ Phân bố Review (Dựa trên độ dài review)
                    if 'CẢM NHẬN' in df_viz.columns:
                        df_viz['Độ sâu Review'] = df_viz['CẢM NHẬN'].apply(lambda x: 'Sâu (Dài)' if len(str(x))>100 else 'Ngắn (Sơ lược)')
                        fig1 = px.pie(df_viz, names='Độ sâu Review', title='Phân bố độ sâu Review', hole=0.3)
                        st.plotly_chart(fig1, use_container_width=True)

    # TAB 2
    with tab2:
        st.header("Dịch Thuật Đa Chiều")
        c1, c2 = st.columns(2)
        with c1:
            txt_in = st.text_area("Nhập văn bản:", height=200)
            if st.button("Dịch Ngay"):
                with st.spinner("Đang xử lý..."):
                    prompt = f"Dịch và phân tích (Việt/Anh/Trung) văn bản: '{txt_in}'"
                    res = model.generate_content(prompt)
                    with c2: st.markdown(res.text)
                    luu_lich_su_vinh_vien("Dịch Thuật", txt_in[:20], res.text)

    # TAB 3
    with tab3:
        st.header("Luyện Tư Duy")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])
        
        if query := st.chat_input("Chủ đề tranh luận..."):
            st.chat_message("user").markdown(query)
            st.session_state.chat_history.append({"role":"user", "content":query})
            
            prompt = f"Phản biện lại: '{query}'"
            res = model.generate_content(prompt)
            
            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append({"role":"assistant", "content":res.text})

    # TAB 4
    with tab4:
        st.header("Kho Lưu Trữ (Google Sheets)")
        if st.button("🔄 Tải lại Lịch sử"):
            st.session_state.history = tai_lich_su_tu_sheet()
            st.rerun()
            
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                with st.expander(f"⏰ {item['time']} | {item['type']} | {item['title']}"):
                    st.markdown(item['content'])
        else:
            st.info("Chưa có lịch sử.")

# --- 6. MAIN ---
def main():
    pm = PasswordManager()
    if not st.session_state.get('user_logged_in', False):
        st.title("🔐 Login System")
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
