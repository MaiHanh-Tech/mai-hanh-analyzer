import streamlit as st
import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document
import numpy as np
import os
from bs4 import BeautifulSoup
import time

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Mai Hanh Strategy (Final)", layout="wide", page_icon="💎")
st.title("💎 The Mai Hanh Analyzer (Final & Clean)")

# Lấy Key từ Secrets
if 'GOOGLE_API_KEY' in st.secrets:
    API_KEY = st.secrets['GOOGLE_API_KEY']
    genai.configure(api_key=API_KEY)
else:
    st.warning("⚠️ Lỗi: Không tìm thấy API Key trong Secrets. Vui lòng kiểm tra lại cấu hình Deployment.")
    st.stop()

# Chọn Model (2.5 Pro là ưu tiên)
try:
    model = genai.GenerativeModel('gemini-2.5-pro') 
except:
    model = genai.GenerativeModel('gemini-2.5-flash')


# --- HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_resource
def load_models():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def doc_noi_dung_file(uploaded_file):
    if not uploaded_file: return ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext == '.pdf':
            reader = PdfReader(uploaded_file)
            return "\\n".join([page.extract_text() for page in reader.pages])
        elif ext == '.docx':
            doc = Document(uploaded_file)
            return "\\n".join([p.text for p in doc.paragraphs])
        elif ext in ['.txt', '.md']:
            return str(uploaded_file.read(), "utf-8")
        elif ext in ['.html', '.htm']:
            soup = BeautifulSoup(uploaded_file, 'html.parser')
            return soup.get_text()
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
            content = [f"{{r['Tên sách']}} {{r['CẢM NHẬN']}}" for i,r in df.iterrows()]
            db_vec = vec_model.encode(content)
            st.success(f"✅ Đã nạp {{len(df)}} cuốn sách cũ.")
        except: st.error("Lỗi file Excel")

st.header("2. Upload Tài Liệu (Chọn nhiều file)")
uploaded_files = st.file_uploader(
    "Kéo thả các file cần phân tích vào đây", 
    type=["pdf","docx","txt","md","html"], 
    accept_multiple_files=True 
)

# NÚT BẤM CHÍNH
if st.button("🚀 PHÂN TÍCH & TỔNG HỢP CHIẾN LƯỢC", type="primary"):
    if not uploaded_files:
        st.warning("Chưa có file nào!")
    else:
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        danh_sach_tom_tat = [] 

        st.subheader("📝 I. Phân Tích Chi Tiết Từng Tài Liệu")
        
        for i, file_doc in enumerate(uploaded_files):
            with st.spinner(f"Đang xử lý file {{i+1}}/{{total_files}}: {{file_doc.name}}..."):
                text = doc_noi_dung_file(file_doc)
                do_dai = len(text)
                
                # --- 1. RAG (LIÊN KẾT) ---
                lien_ket = ""
                if file_excel and len(text) > 50:
                    try:
                        query_vec = vec_model.encode([text[:1000]])
                        scores = cosine_similarity(query_vec, db_vec)[0]
                        top = np.argsort(scores)[::-1][:3]
                        for idx in top:
                            if scores[idx] > 0.35:
                                lien_ket += f"- {{df.iloc[idx]['Tên sách']}}\\n"
                    except: pass
                
                # --- 2. CƠ CHẾ CẮT FILE AN TOÀN (FIX LỖI 429) ---
                text_to_send = text
                ghi_chu_cat = ""
                GIOI_HAN_KY_TU = 30000 
                
                if len(text) > GIOI_HAN_KY TU:
                    text_to_send = text[:15000] + "\\n...\\n" + text[-15000:]
                    ghi_chu_cat = "(Đã phân tích trên Trích đoạn Đầu và Cuối do giới hạn API)"
                
                # --- 3. GỌI API & XỬ LÝ LỖI ---
                try:
                    prompt = f'''
                    Phân tích tài liệu: '{{file_doc.name}}'.
                    Liên kết sách cũ: {{lien_ket}}
                    YÊU CẦU: 1. Tóm tắt cốt lõi. 2. Nhận xét chiều sâu. 3. Trích dẫn câu hay nhất.
                    Nội dung: {{text_to_send}}
                    {ghi_chu_cat}
                    '''
                    res = model.generate_content(prompt)
                    res_text = res.text
                    
                except Exception as e:
                    res_text = f"❌ Lỗi AI: {{e}}.\\n\\n*Mẹo: Vui lòng chờ 1 phút hoặc thử lại file nhỏ hơn.*"

                # HIỂN THỊ VÀ LƯU KẾT QUẢ
                danh_sach_tom_tat.append(f"=== TÀI LIỆU {{i+1}}: {{file_doc.name}} ===\\n{{res_text}}\\n")
                
                with st.expander(f"📄 Kết quả: {{file_doc.name}}", expanded=False):
                    st.markdown(res_text)
            
            progress_bar.progress((i + 1) / total_files)

        # --- GIAI ĐOẠN 2: TỔNG HỢP CHIẾN LƯỢC ---
        st.divider()
        st.header("🏆 II. BÁO CÁO TỔNG QUAN CHIẾN LƯỢC")
        
        if len(danh_sach_tom_tat) > 0:
            with st.spinner("🧠 Đang tổng hợp..."):
                du_lieu_tong_hop = "\\n".join(danh_sach_tom_tat)
                prompt_tong_hop = f'''
                Bạn là Cố vấn Chiến lược. Viết BÁO CÁO TỔNG HỢP (SYNTHESIS) từ các dữ liệu sau:
                {{du_lieu_tong_hop}}
                '''
                
                try:
                    res_tong_hop = model.generate_content(prompt_tong_hop)
                    st.success("Đã hoàn thành!")
                    st.markdown(res_tong_hop.text)
                    st.download_button("💾 Tải Báo Cáo Tổng Hợp", res_tong_hop.text, file_name="Bao_Cao_Tong_Hop.txt")
                except:
                    st.error("Lỗi khi tổng hợp.")


with open("app.py", "w", encoding='utf-8') as f:
    f.write(code_app)

# --- 3. CHẠY NGROK (Khởi động lại) ---
ngrok.set_auth_token(NGROK_TOKEN)
ngrok.kill()
public_url = ngrok.connect(8501).public_url
print(f"\n👉 LINK VIP CỦA CHỊ ĐÂY: {public_url}\n")

!streamlit run app.py
