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

# --- 1. CẤU HÌNH TRANG & SESSION STATE (BỘ NHỚ) ---
st.set_page_config(page_title="Mai Hanh Super App", layout="wide", page_icon="💎")

# Khởi tạo bộ nhớ lịch sử nếu chưa có (Giống ChatGPT)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 2. CẤU HÌNH API & MODEL ---
# Lấy API Key từ Secrets (Cloud) hoặc Sidebar (Local)
if 'GOOGLE_API_KEY' in st.secrets:
    api_key = st.secrets['GOOGLE_API_KEY']
else:
    api_key = st.sidebar.text_input("Nhập Google API Key:", type="password")

if not api_key:
    st.warning("⚠️ Vui lòng nhập API Key để bắt đầu.")
    st.stop()

genai.configure(api_key=api_key)

# CỐ GẮNG DÙNG MODEL MỚI NHẤT (Tháng 12/2025)
try:
    model = genai.GenerativeModel('gemini-2.5-pro') # Giả lập bản tương lai
except:
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-latest') # Bản ổn định
    except:
        model = genai.GenerativeModel('gemini-2.5-flash') # Bản dự phòng

# --- 3. CÁC HÀM XỬ LÝ (BACKEND) ---
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
    """Hàm lưu kết quả vào bộ nhớ tạm"""
    thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "time": thoi_gian,
        "type": loai,
        "title": tieu_de,
        "content": noi_dung
    })

# --- 4. GIAO DIỆN CHÍNH (TABS) ---
st.title("💎 The Mai Hanh Super-App (AI Ecosystem)")

# Tạo các Tab chức năng
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Phân Tích Sách (Analyzer)", 
    "✍️ Dịch Giả Xịn (Linguist)", 
    "🗣️ Tranh Biện (Debater)",
    "aaa Lịch Sử (History)"
])

# ================= TAB 1: PHÂN TÍCH SÁCH (CŨ + NÂNG CẤP) =================
with tab1:
    st.header("Trợ lý Nghiên cứu & Liên kết Tri thức")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        file_excel = st.file_uploader("1. Kết nối Kho Sách (Excel)", type="xlsx", key="tab1_excel")
        uploaded_files = st.file_uploader("2. Upload Tài liệu mới", type=["pdf","docx","txt","md"], accept_multiple_files=True, key="tab1_files")
        btn_analyze = st.button("🚀 Phân Tích Chiến Lược", type="primary")

    with col_b:
        if btn_analyze and uploaded_files:
            # Setup Vector
            vec_model = None
            db_vec = None
            df = None
            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
                    vec_model = load_models()
                    content = [f"{r['Tên sách']} {r['CẢM NHẬN']}" for i,r in df.iterrows()]
                    db_vec = vec_model.encode(content)
                    st.success(f"✅ Đã kết nối {len(df)} cuốn sách cũ.")
                except: st.error("Lỗi file Excel")

            # Xử lý từng file
            full_report = ""
            progress = st.progress(0)
            
            for i, file_doc in enumerate(uploaded_files):
                text = doc_file(file_doc)
                
                # RAG
                lien_ket = ""
                if file_excel and len(text) > 100:
                    try:
                        query_vec = vec_model.encode([text[:1000]])
                        scores = cosine_similarity(query_vec, db_vec)[0]
                        top = np.argsort(scores)[::-1][:3]
                        for idx in top:
                            if scores[idx] > 0.35:
                                lien_ket += f"- {df.iloc[idx]['Tên sách']}\n"
                    except: pass
                
                prompt = f"""
                Phân tích tài liệu: '{file_doc.name}'.
                Liên kết sách cũ: {lien_ket}
                Yêu cầu: Tóm tắt, Nhận xét sâu sắc, Trích dẫn hay.
                Nội dung: {text}
                """
                res = model.generate_content(prompt)
                
                with st.expander(f"📄 Kết quả: {file_doc.name}", expanded=True):
                    st.markdown(res.text)
                
                full_report += f"=== TÀI LIỆU: {file_doc.name} ===\n{res.text}\n\n"
                progress.progress((i+1)/len(uploaded_files))
            
            # Tổng hợp
            if len(uploaded_files) > 1:
                with st.spinner("Đang tổng hợp chiến lược..."):
                    prompt_syn = f"Tổng hợp chiến lược từ các báo cáo sau:\n{full_report}"
                    res_syn = model.generate_content(prompt_syn)
                    st.success("🏆 BÁO CÁO TỔNG HỢP")
                    st.markdown(res_syn.text)
                    full_report = f"BÁO CÁO TỔNG HỢP:\n{res_syn.text}\n\n" + full_report
            
            # Lưu vào lịch sử
            luu_lich_su("Phân Tích Sách", f"Batch {len(uploaded_files)} files", full_report)

# ===== TAB 2: DỊCH GIẢ ĐA NĂNG (NÂNG CẤP) =====
with tab2:
    st.header("Dịch Thuật Thông Minh (Tự động nhận diện)")
    c1, c2 = st.columns(2)
    with c1:
        txt_in = st.text_area("Nhập văn bản (Việt/Anh/Trung):", height=300, placeholder="Ví dụ: Nhập tiếng Việt để dịch sang Anh & Trung. Nhập ngoại ngữ để dịch sang Việt.")
        style = st.selectbox("Phong cách:", ["Hàn lâm/Học thuật", "Văn học/Cảm xúc", "Đời thường/Dễ hiểu", "Thương mại/Kinh tế"])
        if st.button("✍️ Dịch Ngay"):
            if txt_in:
                with st.spinner("Đang phân tích ngôn ngữ và dịch..."):
                    # PROMPT THÔNG MINH ĐA NGÔN NGỮ
                    prompt = f"""
                    Bạn là Chuyên gia Ngôn ngữ cao cấp.
                    Nhiệm vụ: Dịch và Phân tích văn bản sau.
                    
                    INPUT: "{txt_in}"
                    
                    LOGIC XỬ LÝ:
                    1. Tự động nhận diện ngôn ngữ đầu vào.
                    2. **NẾU LÀ TIẾNG VIỆT**:
                       - Dịch sang **Tiếng Anh** (Phong cách: {style}).
                       - Dịch sang **Tiếng Trung** (Bao gồm: Chữ Hán, **Pinyin**, và Hán Việt).
                    3. **NẾU LÀ NGOẠI NGỮ (Anh/Trung/Pháp...)**:
                       - Dịch sang **Tiếng Việt** (Phong cách: {style}).
                    
                    YÊU CẦU BỔ SUNG:
                    - Sau khi dịch, hãy chọn ra 3 từ vựng/cấu trúc ngữ pháp đắt giá nhất để phân tích sâu (ngữ nghĩa, cách dùng).
                    - Trình bày rõ ràng, dễ nhìn.
                    """
                    res = model.generate_content(prompt)
                    with c2:
                        st.markdown(res.text)
                    luu_lich_su("Dịch Thuật", f"Dịch: {{txt_in[:20]}}...", res.text)

# ================= TAB 3: TRANH BIỆN (DEBATER - MỚI) =================
with tab3:
    st.header("Luyện Tư Duy & Phản Biện")
    
    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input chat
    if user_query := st.chat_input("Nhập chủ đề muốn tranh luận (VD: AI có thay thế con người?)..."):
        # Hiển thị câu hỏi của user
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # AI trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đối thủ đang suy nghĩ..."):
                # Gửi kèm lịch sử chat để nó nhớ ngữ cảnh
                history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
                
                prompt = f"""
                Bạn là một Giáo sư/Triết gia phản biện khó tính.
                Nhiệm vụ: Tranh luận với người dùng về chủ đề này để giúp họ rèn luyện tư duy.
                
                Lịch sử cuộc trò chuyện:
                {history_context}
                
                Người dùng vừa nói: "{user_query}"
                
                Hãy phản bác lại, hoặc đặt câu hỏi sâu sắc để người dùng phải suy nghĩ lại quan điểm của mình. Đừng đồng ý quá dễ dàng.
                """
                response = model.generate_content(prompt)
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})

# ================= TAB 4: LỊCH SỬ (MEMORY) =================
with tab4:
    st.header("aaa Kho Lưu Trữ Tác Vụ")
    st.caption("Lưu trữ tạm thời trong phiên làm việc này. Refresh trang sẽ mất.")
    
    if len(st.session_state.history) == 0:
        st.info("Chưa có dữ liệu lịch sử.")
    else:
        for item in reversed(st.session_state.history):
            with st.expander(f"⏰ {item['time']} | {item['type']} | {item['title']}"):
                st.markdown(item['content'])
                st.download_button("Tải về", item['content'], file_name=f"Log_{item['time']}.txt")
