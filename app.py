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
import plotly.express as px # Thư viện vẽ biểu đồ

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

# --- 3. DATABASE MANAGER (GOOGLE SHEETS) ---
def connect_gsheet():
    try:
        if "gcp_service_account" not in st.secrets:
            return None

        # Lấy thông tin và FIX LỖI KHÓA
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n").replace('\\n', '\n')

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
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

# --- 4. CÁC HÀM XỬ LÝ AI & FILE ---
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

# --- 5. GIAO DIỆN CHÍNH (ĐÃ SỬA TAB 1 & TAB 2) ---
def show_main_app():
    # Load history
    if 'history_loaded' not in st.session_state:
        cloud_data = tai_lich_su_tu_sheet()
        if cloud_data: st.session_state.history = cloud_data
        st.session_state.history_loaded = True
    
    if 'history' not in st.session_state: st.session_state.history = []
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []

    # Cấu hình Gemini Thông Minh (Tự động chọn Model)
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        # Thử lần lượt các model
        try:
            model = genai.GenerativeModel('gemini-2.5-pro')
        except:
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
            except:
                model = genai.GenerativeModel('gemini-pro')
    except:
        st.error("❌ Lỗi: Chưa cấu hình [system] gemini_api_key trong Secrets!")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.success(f"👤 User: {st.session_state.current_user_name}")
        if st.button("Đăng Xuất"):
            st.session_state.user_logged_in = False
            st.rerun()

    st.title("💎 The Mai Hanh Super-App")
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Phân Tích Sách", "✍️ Dịch Giả", "🗣️ Tranh Biện", "⏳ Lịch Sử"])

    # === TAB 1: PHÂN TÍCH SÁCH (FULL WIDTH + BIỂU ĐỒ) ===
    with tab1:
        st.header("Trợ lý Nghiên cứu RAG")
        
        # Phần Upload (Gọn gàng)
        with st.container():
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                file_excel = st.file_uploader("1. Kết nối Kho Sách", type="xlsx", key="tab1_excel")
            with c2:
                uploaded_files = st.file_uploader("2. Tài liệu mới", type=["pdf","docx","txt","md","html"], accept_multiple_files=True)
            with c3:
                st.write("")
                st.write("")
                btn_run = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)
        
        st.divider()

        # Phần Xử lý & Kết quả
        if btn_run and uploaded_files:
            vec_model = load_models()
            db_vec, df = None, None
            has_db = False
            
            # Xử lý Excel
            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
                    content = [f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for i,r in df.iterrows()]
                    db_vec = vec_model.encode(content)
                    has_db = True
                    st.success(f"✅ Đã kết nối {len(df)} cuốn sách từ kho dữ liệu.")
                except: st.error("Lỗi đọc Excel.")

            # Chạy từng file
            for f in uploaded_files:
                text = doc_file(f)
                lien_ket = ""
                if has_db:
                    q_vec = vec_model.encode([text[:2000]])
                    scores = cosine_similarity(q_vec, db_vec)[0]
                    top = np.argsort(scores)[::-1][:3]
                    for idx in top:
                        if scores[idx] > 0.35:
                            lien_ket += f"- {df.iloc[idx]['Tên sách']} (Khớp: {scores[idx]*100:.1f}%)\n"
                
                with st.spinner(f"Đang phân tích {f.name}..."):
                    prompt = f"Phân tích tài liệu '{f.name}'. Liên kết cũ: {lien_ket}. Nội dung: {text[:20000]}"
                    res = model.generate_content(prompt)
                    
                    st.markdown(f"### 📄 Kết quả: {f.name}")
                    st.markdown(res.text)
                    st.markdown("---")
                    luu_lich_su_vinh_vien("Phân Tích", f.name, res.text)

        # Phần Biểu đồ (Luôn hiện nếu có Excel)
        if file_excel:
            try:
                if 'df_viz' not in st.session_state:
                    st.session_state.df_viz = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
                df_v = st.session_state.df_viz
                
                with st.expander("📊 Thống Kê Kho Sách", expanded=True):
                    g1, g2 = st.columns(2)
                    with g1:
                        if 'Tác giả' in df_v.columns:
                            top_auth = df_v['Tác giả'].value_counts().head(10).reset_index()
                            top_auth.columns = ['Tác giả', 'Số lượng']
                            st.plotly_chart(px.bar(top_auth, x='Số lượng', y='Tác giả', orientation='h', title="Top Tác giả"), use_container_width=True)
                    with g2:
                        if 'CẢM NHẬN' in df_v.columns:
                            df_v['Len'] = df_v['CẢM NHẬN'].apply(lambda x: len(str(x)))
                            st.plotly_chart(px.histogram(df_v, x='Len', title="Độ sâu Review"), use_container_width=True)
            except: pass

    # === TAB 2: DỊCH GIẢ (KHÔNG CHIA CỘT + DOWNLOAD HTML) ===
    with tab2:
        st.header("Dịch Thuật Đa Chiều")
        
        # Input full width
        txt_in = st.text_area("Nhập văn bản cần dịch (Tự động nhận diện ngôn ngữ):", height=150)
        
        if st.button("✍️ Dịch & Phân Tích Ngay", type="primary"):
            if txt_in:
                with st.spinner("AI đang tư duy..."):
                    prompt = f"""
                    Bạn là Chuyên gia Ngôn ngữ. Hãy xử lý văn bản sau: "{txt_in}"
                    
                    YÊU CẦU:
                    1. Nếu là Tiếng Việt -> Dịch sang Tiếng Anh (Hàn lâm) và Tiếng Trung (Kèm Pinyin).
                    2. Nếu là Ngoại ngữ -> Dịch sang Tiếng Việt (Văn phong mượt mà).
                    3. Phân tích 3 từ vựng/cấu trúc ngữ pháp đắt giá nhất trong văn bản.
                    
                    TRÌNH BÀY: Dùng Markdown rõ ràng.
                    """
                    res = model.generate_content(prompt)
                    
                    # Hiện kết quả Full Width
                    st.markdown("### 🎯 Kết Quả:")
                    st.markdown(res.text)
                    
                    # Tạo nội dung HTML để download
                    html_content = f"""
                    <html>
                    <head><style>body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; }}</style></head>
                    <body>
                        <h2>Bản Dịch & Phân Tích</h2>
                        <div style="background: #f0f2f6; padding: 15px; border-radius: 5px;">
                            <strong>Gốc:</strong><br>{txt_in}
                        </div>
                        <hr>
                        {markdown.markdown(res.text)} <!-- Cần import markdown nếu muốn đẹp hơn, hoặc để text thô -->
                    </body>
                    </html>
                    """
                    # Nút Download
                    st.download_button(
                        label="💾 Tải kết quả (HTML)",
                        data=html_content,
                        file_name="Ban_Dich.html",
                        mime="text/html"
                    )
                    
                    luu_lich_su_vinh_vien("Dịch Thuật", txt_in[:30], res.text)
            else:
                st.warning("Vui lòng nhập văn bản!")

    # === TAB 3: TRANH BIỆN ===
    with tab3:
        st.header("Luyện Tư Duy")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])
        
        if query := st.chat_input("Chủ đề tranh luận..."):
            st.chat_message("user").markdown(query)
            st.session_state.chat_history.append({"role":"user", "content":query})
            
            prompt = f"Phản biện lại quan điểm này: '{query}'"
            res = model.generate_content(prompt)
            
            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append({"role":"assistant", "content":res.text})

    # === TAB 4: LỊCH SỬ ===
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
        st.title("🔐 Mai Hạnh Login")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            user_pass = st.text_input("Password:", type="password")
            if st.button("Login", use_container_width=True):
                if pm.check_password(user_pass):
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = user_pass
                    st.session_state.current_user_name = st.session_state.key_name_mapping.get(user_pass, "User")
                    st.rerun()
                else: st.error("Sai mật khẩu!")
    else:
        show_main_app()

if __name__ == "__main__":
    import markdown # Import thêm ở đây để dùng cho nút Download HTML
    main()
