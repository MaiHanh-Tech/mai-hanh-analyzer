
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

# --- CẤU HÌNH ---
st.set_page_config(page_title="Mai Hanh Strategy", layout="wide", page_icon="💎")
st.title("💎 The Mai Hanh Analyzer (Cloud Version)")

# --- QUẢN LÝ BẢO MẬT (SECRETS) ---
# Khi lên Cloud, API Key sẽ được lấy từ hệ thống bảo mật của Streamlit
# Chứ không dán cứng vào code để tránh bị lộ
if 'GOOGLE_API_KEY' in st.secrets:
    api_key = st.secrets['GOOGLE_API_KEY']
else:
    api_key = st.sidebar.text_input("Nhập Google API Key:", type="password")

if not api_key:
    st.warning("⚠️ Vui lòng nhập API Key để tiếp tục.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# --- HÀM XỬ LÝ ---
@st.cache_resource
def load_models():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def doc_noi_dung_file(uploaded_file):
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
            text = soup.get_text()
            return text
    except: return ""
    return ""

# --- GIAO DIỆN ---
with st.sidebar:
    st.header("1. Kết Nối Kho Sách")
    file_excel = st.file_uploader("Upload Book146.xlsx", type="xlsx")
    
    vec_model = None
    db_vec = None
    df = None
    
    if file_excel:
        try:
            df = pd.read_excel(file_excel).dropna(subset=['Tên sách'])
            vec_model = load_models()
            content = [f"{r['Tên sách']} {r['CẢM NHẬN']}" for i,r in df.iterrows()]
            db_vec = vec_model.encode(content)
            st.success(f"✅ Đã nạp {len(df)} cuốn sách cũ.")
        except: st.error("Lỗi file Excel")

st.header("2. Upload Tài Liệu (Chọn nhiều file)")
uploaded_files = st.file_uploader(
    "Kéo thả các file cần phân tích vào đây", 
    type=["pdf","docx","txt","md","html"], 
    accept_multiple_files=True 
)

if st.button("🚀 PHÂN TÍCH & TỔNG HỢP CHIẾN LƯỢC", type="primary"):
    if not uploaded_files:
        st.warning("Chưa có file nào!")
    else:
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        danh_sach_tom_tat = [] 

        st.subheader("📝 I. Phân Tích Chi Tiết")
        
        for i, file_doc in enumerate(uploaded_files):
            with st.spinner(f"Đang đọc file {i+1}/{total_files}: {file_doc.name}..."):
                text = doc_noi_dung_file(file_doc)
                
                # RAG
                lien_ket = ""
                if file_excel and len(text) > 50:
                    try:
                        query_vec = vec_model.encode([text[:1000]])
                        scores = cosine_similarity(query_vec, db_vec)[0]
                        top = np.argsort(scores)[::-1][:3]
                        for idx in top:
                            if scores[idx] > 0.35:
                                lien_ket += f"- {df.iloc[idx]['Tên sách']}\n"
                    except: pass

                # Prompt
                prompt = f'''
                Phân tích tài liệu: '{file_doc.name}'.
                Liên kết sách cũ: {lien_ket}
                Yêu cầu: Tóm tắt, Nhận xét sâu sắc, Trích dẫn hay.
                Nội dung: {text}
                '''
                
                try:
                    res = model.generate_content(prompt)
                    danh_sach_tom_tat.append(f"=== TÀI LIỆU {i+1}: {file_doc.name} ===\n{res.text}\n")
                    with st.expander(f"📄 Kết quả: {file_doc.name}", expanded=False):
                        st.markdown(res.text)
                except Exception as e:
                    st.error(f"Lỗi AI: {e}")
            
            progress_bar.progress((i + 1) / total_files)

        st.divider()
        st.header("🏆 II. BÁO CÁO TỔNG QUAN CHIẾN LƯỢC")
        
        if len(danh_sach_tom_tat) > 0:
            with st.spinner("🧠 Đang tổng hợp..."):
                du_lieu_tong_hop = "\n".join(danh_sach_tom_tat)
                prompt_tong_hop = f'''
                Bạn là Cố vấn Chiến lược. Viết BÁO CÁO TỔNG HỢP (SYNTHESIS) từ:
                {du_lieu_tong_hop}
                '''
                try:
                    res_tong_hop = model.generate_content(prompt_tong_hop)
                    st.success("Đã hoàn thành!")
                    st.markdown(res_tong_hop.text)
                except: st.error("Lỗi tổng hợp.")
