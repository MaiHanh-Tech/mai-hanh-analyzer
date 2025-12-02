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

   # --- CẤU HÌNH GEMINI (LOGIC CHỐNG SẬP APP) ---
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        
        # 1. THỬ BẢN MỚI NHẤT & MẠNH NHẤT (PRO)
        try:
            model = genai.GenerativeModel('gemini-2.5-pro')
            st.sidebar.success("🤖 Lõi: Gemini 2.5 Pro (Cao cấp)")
        except:
            # 2. THỬ BẢN DỰ PHÒNG TỐC ĐỘ (FLASH)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                st.sidebar.info("🤖 Lõi: Gemini 2.5 Flash (Tốc độ ổn định)")
            except:
                # 3. DÙNG BẢN LÂU ĐỜI NHẤT (BẮT BUỘC PHẢI CÓ)
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash') # Giả định model này có
                    st.sidebar.warning("🤖 Lõi: Gemini 2.5 Flash (Dự phòng)")
                except:
                    model = genai.GenerativeModel('gemini-pro') # Model cũ nhưng mạnh
                    st.sidebar.error("🤖 Lõi: Gemini Pro (Lõi cũ)")
    
    except Exception as e:
        st.error(f"❌ Lỗi: Chưa cấu hình [system] gemini_api_key trong Secrets!")
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
        st.header("📚 Trợ lý Nghiên cứu RAG")
        
        # --- PHẦN 1: UPLOAD & CẤU HÌNH (ĐỂ TRÊN CÙNG) ---
        with st.container():
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                file_excel = st.file_uploader("1. Kết nối Kho Sách (Excel)", type="xlsx", key="tab1_excel")
            with c2:
                uploaded_files = st.file_uploader("2. Tài liệu mới cần đọc", type=["pdf","docx","txt", "md", "html"], accept_multiple_files=True)
            with c3:
                st.write("") # Spacer
                st.write("")
                btn_run = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)

        st.divider() # Đường kẻ ngang phân cách

        # --- PHẦN 2: XỬ LÝ & HIỂN THỊ (TRÀN MÀN HÌNH) ---
        if btn_run and uploaded_files:
            # Load Model
            with st.spinner("Đang khởi động bộ não Vector..."):
                vec_model = load_models()
                db_vec, df = None, None
                has_db = False
                
                # Xử lý Excel
                if file_excel:
                    try:
                        df = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
                        if not df.empty:
                            content = [f"{r['Tên sách']} {str(r['CẢM NHẬN'])}" for i,r in df.iterrows()]
                            db_vec = vec_model.encode(content)
                            has_db = True
                            st.success(f"✅ Đã kết nối {len(df)} cuốn sách từ kho dữ liệu.")
                    except: st.error("❌ Lỗi đọc file Excel.")

            # Chạy từng file
            for f in uploaded_files:
                with st.status(f"🤖 Đang đọc và phân tích: {f.name}...", expanded=True) as status:
                    text = doc_file(f)
                    st.write(f"Đã đọc {len(text)} ký tự.")
                    
                    lien_ket = ""
                    # RAG Logic
                    if has_db:
                        st.write("Đang tìm liên kết trong kho sách cũ...")
                        q_vec = vec_model.encode([text[:20000]]) 
                        scores = cosine_similarity(q_vec, db_vec)[0]
                        top = np.argsort(scores)[::-1][:3]
                        for idx in top:
                            if scores[idx] > 0.40:
                                lien_ket += f"- {df.iloc[idx]['Tên sách']} (Khớp: {scores[idx]*100:.1f}%)\n"
                    
                    # Gemini
                    st.write("Đang viết báo cáo...")
                    prompt = f"Phân tích tài liệu '{f.name}'. Nguồn liên kết cũ: {lien_ket}. Nội dung: {text[:20000]}"
                    res = model.generate_content(prompt)
                    
                    status.update(label="✅ Hoàn tất!", state="complete", expanded=False)

                # HIỂN THỊ KẾT QUẢ (FULL WIDTH)
                st.markdown(f"### 📄 Kết quả: {f.name}")
                st.markdown(res.text)
                st.markdown("---")
                
                # Lưu lịch sử
                luu_lich_su_vinh_vien("Phân Tích", f.name, res.text)

        # --- PHẦN 3: BIỂU ĐỒ (HIỆN LUÔN KHÔNG CẦN BẤM) ---
        if file_excel:
            try:
                # Đọc lại file để vẽ (nếu chưa có trong session)
                if 'df_viz' not in st.session_state:
                    st.session_state.df_viz = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
                
                df_v = st.session_state.df_viz
                
                st.subheader("📊 Bản Đồ Kho Sách Của Chị")
                import plotly.express as px
                
                # Chia 2 cột cho biểu đồ
                g1, g2 = st.columns(2)
                
                with g1:
                    # Biểu đồ Tác giả
                    if 'Tác giả' in df_v.columns:
                        top_auth = df_v['Tác giả'].value_counts().head(10).reset_index()
                        top_auth.columns = ['Tác giả', 'Số lượng']
                        fig = px.bar(top_auth, x='Số lượng', y='Tác giả', orientation='h', title="Top Tác giả yêu thích")
                        st.plotly_chart(fig, use_container_width=True)
                
                with g2:
                    # Biểu đồ Review (Giả lập độ sâu)
                    if 'CẢM NHẬN' in df_v.columns:
                        df_v['Độ dài'] = df_v['CẢM NHẬN'].apply(lambda x: len(str(x)))
                        fig2 = px.histogram(df_v, x='Độ dài', title="Phân bố độ sâu Review (Độ dài chữ)")
                        st.plotly_chart(fig2, use_container_width=True)
                        
            except Exception as e:
                st.warning(f"Chưa thể vẽ biểu đồ: {e}")

    # TAB 2
    with tab2:
        st.header("Dịch Thuật Đa Chiều")
        c1, c2 = st.columns(2)
        with c1:
            txt_in = st.text_area("Nhập văn bản (Việt/Anh/Trung):", height=200)
            txt_in = st.text_area("Nhập văn bản:", height=200)
            if st.button("Dịch Ngay"):
                with st.spinner("Đang xử lý..."):
                    prompt = f"""
                    Bạn là Chuyên gia Ngôn ngữ. Xử lý văn bản: "{txt_in}"
                    Logic:
                    - Nếu là Tiếng Việt -> Dịch sang Anh & Trung (kèm Pinyin).
                    - Nếu là Ngoại ngữ -> Dịch sang Tiếng Việt (Văn phong hay).
                    - Phân tích 3 từ vựng hay nhất.
                    """
                    prompt = f"Dịch và phân tích (Việt/Anh/Trung) cho văn bản: '{txt_in}'"
                    res = model.generate_content(prompt)
                    with c2: st.markdown(res.text)
                    luu_lich_su("Dịch Thuật", txt_in[:20], res.text)
                    # LƯU VĨNH VIỄN
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
