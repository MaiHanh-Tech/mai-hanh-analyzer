import streamlit as st
import google.generativeai as genai
import asyncio 
import time    
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
import plotly.express as px
import markdown
import edge_tts
import asyncio
import json
import re
from streamlit_agraph import agraph, Node, Edge, Config
import sys
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# Fix lỗi asyncio trên Windows (nếu chạy local)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="The Cognitive Weaver", layout="wide", page_icon="💎")

# ==========================================
# 🌍 BỘ TỪ ĐIỂN ĐA NGÔN NGỮ (I18N)
# ==========================================
TRANS = {
    "vi": {
        "title": "🕸️ Người Dệt Nhận Thức",
        "login_title": "🔐 Đăng Nhập Hệ Thống",
        "login_btn": "Đăng Nhập",
        "pass_placeholder": "Nhập mật khẩu truy cập...",
        "wrong_pass": "Sai mật khẩu!",
        "logout": "Đăng Xuất",
        "welcome": "Xin chào",
        "role_admin": "Quản Trị Viên",
        "role_user": "Thành Viên",
        "lang_select": "Ngôn ngữ / Language / 语言",
        # Tabs
        "tab1": "📚 Phân Tích Sách",
        "tab2": "✍️ Dịch Giả",
        "tab3": "🗣️ Tranh Biện",
        "tab4": "🎙️ Phòng Thu AI",
        "tab5": "⏳ Nhật Ký",
        # Tab 1
        "t1_header": "Trợ lý Nghiên cứu & Knowledge Graph",
        "t1_up_excel": "1. Kết nối Kho Sách (Excel)",
        "t1_up_doc": "2. Tài liệu mới (PDF/Docx)",
        "t1_btn": "🚀 PHÂN TÍCH NGAY",
        "t1_connect_ok": "✅ Đã kết nối {n} cuốn sách.",
        "t1_analyzing": "Đang phân tích {name}...",
        "t1_graph_title": "🪐 Vũ Trụ Sách",
        # Tab 2
        "t2_header": "Dịch Thuật Đa Chiều",
        "t2_input": "Nhập văn bản cần dịch:",
        "t2_target": "Dịch sang:",
        "t2_style": "Phong cách:",
        "t2_btn": "✍️ Dịch Ngay",
        "t2_styles": ["Mặc định", "Hàn lâm/Học thuật", "Văn học/Cảm xúc", "Đời thường", "Kinh tế", "Kiếm hiệp"],
        # Tab 3
        "t3_header": "Đấu Trường Tư Duy",
        "t3_persona_label": "Chọn Đối Thủ:",
        "t3_input": "Nhập chủ đề tranh luận...",
        "t3_clear": "🗑️ Xóa Chat",
        # Tab 4
        "t4_header": "🎙️ Phòng Thu AI Đa Ngôn Ngữ",
        "t4_voice": "Chọn Giọng:",
        "t4_speed": "Tốc độ:",
        "t4_btn": "🔊 TẠO AUDIO",
        "t4_dl": "⬇️ TẢI MP3",
        # Tab 5
        "t5_header": "Nhật Ký & Lịch Sử",
        "t5_refresh": "🔄 Tải lại Lịch sử",
        "t5_empty": "Chưa có dữ liệu lịch sử.",
        "t5_chart": "📈 Biểu đồ Cảm xúc",
    },
    "en": {
        "title": "🕸️ The Cognitive Weaver",
        "login_title": "🔐 System Login",
        "login_btn": "Login",
        "pass_placeholder": "Enter password...",
        "wrong_pass": "Wrong password!",
        "logout": "Logout",
        "welcome": "Welcome",
        "role_admin": "Admin",
        "role_user": "Member",
        "lang_select": "Language",
        # Tabs
        "tab1": "📚 Book Analysis",
        "tab2": "✍️ Translator",
        "tab3": "🗣️ Debater",
        "tab4": "🎙️ AI Studio",
        "tab5": "⏳ History",
        # Tab 1
        "t1_header": "Research Assistant & Knowledge Graph",
        "t1_up_excel": "1. Connect Book Database (Excel)",
        "t1_up_doc": "2. New Documents (PDF/Docx)",
        "t1_btn": "🚀 ANALYZE NOW",
        "t1_connect_ok": "✅ Connected {n} books.",
        "t1_analyzing": "Analyzing {name}...",
        "t1_graph_title": "🪐 Book Universe",
        # Tab 2
        "t2_header": "Multidimensional Translator",
        "t2_input": "Enter text to translate:",
        "t2_target": "Translate to:",
        "t2_style": "Style:",
        "t2_btn": "✍️ Translate",
        "t2_styles": ["Default", "Academic", "Literary/Emotional", "Casual", "Business", "Wuxia/Martial Arts"],
        # Tab 3
        "t3_header": "Thinking Arena",
        "t3_persona_label": "Choose Opponent:",
        "t3_input": "Enter debate topic...",
        "t3_clear": "🗑️ Clear Chat",
        # Tab 4
        "t4_header": "🎙️ Multilingual AI Studio",
        "t4_voice": "Select Voice:",
        "t4_speed": "Speed:",
        "t4_btn": "🔊 GENERATE AUDIO",
        "t4_dl": "⬇️ DOWNLOAD MP3",
        # Tab 5
        "t5_header": "Logs & History",
        "t5_refresh": "🔄 Refresh History",
        "t5_empty": "No history data found.",
        "t5_chart": "📈 Emotion Chart",
    },
    "zh": {
        "title": "🕸️ 认知编织者 (The Cognitive Weaver)",
        "login_title": "🔐 系统登录",
        "login_btn": "登录",
        "pass_placeholder": "请输入密码...",
        "wrong_pass": "密码错误！",
        "logout": "登出",
        "welcome": "你好",
        "role_admin": "管理员",
        "role_user": "成员",
        "lang_select": "语言",
        # Tabs
        "tab1": "📚 书籍分析",
        "tab2": "✍️ 翻译专家",
        "tab3": "🗣️ 辩论场",
        "tab4": "🎙️ AI 录音室",
        "tab5": "⏳ 历史记录",
        # Tab 1
        "t1_header": "研究助手 & 知识图谱",
        "t1_up_excel": "1. 连接书库 (Excel)",
        "t1_up_doc": "2. 上传新文档 (PDF/Docx)",
        "t1_btn": "🚀 立即分析",
        "t1_connect_ok": "✅ 已连接 {n} 本书。",
        "t1_analyzing": "正在分析 {name}...",
        "t1_graph_title": "🪐 书籍宇宙",
        # Tab 2
        "t2_header": "多维翻译",
        "t2_input": "输入文本:",
        "t2_target": "翻译成:",
        "t2_style": "风格:",
        "t2_btn": "✍️ 翻译",
        "t2_styles": ["默认", "学术", "文学/情感", "日常", "商业", "武侠"],
        # Tab 3
        "t3_header": "思维竞技场",
        "t3_persona_label": "选择对手:",
        "t3_input": "输入辩论主题...",
        "t3_clear": "🗑️ 清除聊天",
        # Tab 4
        "t4_header": "🎙️ AI 多语言录音室",
        "t4_voice": "选择声音:",
        "t4_speed": "语速:",
        "t4_btn": "🔊 生成音频",
        "t4_dl": "⬇️ 下载 MP3",
        # Tab 5
        "t5_header": "日志 & 历史",
        "t5_refresh": "🔄 刷新历史",
        "t5_empty": "暂无历史数据。",
        "t5_chart": "📈 情绪图表",
    }
}

# --- 2. CLASS QUẢN LÝ MẬT KHẨU ---
class PasswordManager:
    def __init__(self):
        self.user_tiers = st.secrets.get("user_tiers", {})
        if "key_name_mapping" not in st.session_state:
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
        # Sửa: Cho phép admin_password HOẶC pass "admin_maihanh" là Admin
        is_master = (password == st.secrets.get("admin_password"))
        is_maihanh = (password == "admin_maihanh") 
        return is_master or is_maihanh

# --- HÀM GỌI API AN TOÀN (FIX LỖI QUOTA) ---
def run_gemini_safe(model_func, prompt, retries=3):
    """Hàm bọc để tự động chờ khi hết quota"""
    for i in range(retries):
        try:
            return model_func(prompt)
        except ResourceExhausted:
            wait_time = (i + 1) * 10  
            st.warning(f"⚠️ Hệ thống đang bận (Hết Quota), đang thử lại sau {wait_time}s... (Lần {i+1}/{retries})")
            time.sleep(wait_time)
        except Exception as e:
            time.sleep(5)
            if i == retries - 1: return None
    st.error("❌ Hệ thống quá tải. Vui lòng thử lại sau vài phút.")
    return None

# --- 3. DATABASE MANAGER ---
def connect_gsheet():
    try:
        if "gcp_service_account" not in st.secrets: return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n").replace('\\n', '\n')
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("AI_History_Logs").sheet1
    except: return None

# --- 3b. SENTIMENT ANALYSIS ---
def phan_tich_cam_xuc(text: str):
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        
        # --- CẤU HÌNH MODEL CHO PHÂN TÍCH CẢM XÚC (Pro trước, Flash sau) ---
        try: 
            model = genai.GenerativeModel("gemini-2.5-pro")
        except: 
            model = genai.GenerativeModel("gemini-2.5-flash")
        # -------------------------------------------------------------------

        prompt = f"""Analyze sentiment. Return JSON: {{"sentiment_score": float (-1.0 to 1.0), "sentiment_label": string}}. Text: \"\"\"{text[:1000]}\"\"\""""
        try:
            res = model.generate_content(prompt)
            m = re.search(r"\{.*\}", res.text, re.S)
            if m:
                data = json.loads(m.group(0))
                return float(data.get("sentiment_score", 0)), str(data.get("sentiment_label", "Neutral"))
        except: pass
    except: pass
    return 0.0, "Neutral"

# --- LƯU & TẢI ---
def luu_lich_su_vinh_vien(loai, tieu_de, noi_dung):
    thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = st.session_state.get("current_user_name", "Unknown")
    score, label = 0.0, "Neutral"
    if len(noi_dung) > 10 and "{" not in noi_dung[:5]:
         score, label = phan_tich_cam_xuc(tieu_de + ": " + noi_dung)

    if "history" not in st.session_state: st.session_state.history = []
    st.session_state.history.append({
        "time": thoi_gian, "type": loai, "title": tieu_de, "content": noi_dung,
        "user": current_user, "sentiment_score": score, "sentiment_label": label,
    })

    try:
        sheet = connect_gsheet()
        if sheet: sheet.append_row([thoi_gian, loai, tieu_de, noi_dung, current_user, score, label])
    except: pass

def tai_lich_su_tu_sheet():
    try:
        sheet = connect_gsheet()
        if sheet:
            data = sheet.get_all_records()
            formatted = []
            my_user = st.session_state.get("current_user_name", "")
            i_am_admin = st.session_state.get("is_admin", False)

            for item in data:
                row_owner = item.get("User", "Unknown")
                if i_am_admin or (row_owner == my_user):
                    formatted.append({
                        "time": item.get("Time", ""), "type": item.get("Type", ""),
                        "title": item.get("Title", ""), "content": item.get("Content", ""),
                        "user": row_owner, "sentiment_score": item.get("SentimentScore", 0.0),
                        "sentiment_label": item.get("SentimentLabel", "Neutral"),
                    })
            return formatted
    except: return []
    return []

# --- 4. CÁC HÀM XỬ LÝ KHÁC ---
@st.cache_resource
def load_models():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def doc_file(uploaded_file):
    if not uploaded_file: return ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif ext == ".docx":
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".txt", ".md"]:
            return str(uploaded_file.read(), "utf-8")
        elif ext in [".html", ".htm"]:
            soup = BeautifulSoup(uploaded_file, "html.parser")
            return soup.get_text()
    except: return ""
    return ""

def generate_edge_audio_sync(text, voice_code, rate, out_path="studio_output.mp3"):
    async def _gen():
        communicate = edge_tts.Communicate(text, voice_code, rate=rate)
        await communicate.save(out_path)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_gen()); import time; time.sleep(2)
        else: loop.run_until_complete(_gen())
    except: asyncio.run(_gen())

# --- 5. GIAO DIỆN CHÍNH ---
def show_main_app():
    # Load History
    if "history_loaded" not in st.session_state:
        cloud_data = tai_lich_su_tu_sheet()
        if cloud_data: st.session_state.history = cloud_data
        st.session_state.history_loaded = True
    if "history" not in st.session_state: st.session_state.history = []
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    # Config Gemini
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        
        # --- [ĐÃ SỬA] ƯU TIÊN PRO TRƯỚC, FLASH SAU ---
        try: 
            model = genai.GenerativeModel("gemini-2.5-pro")
        except: 
            model = genai.GenerativeModel("gemini-2.5-flash")
        # ----------------------------------------------
            
    except: st.stop()

    # --- SIDEBAR & NGÔN NGỮ ---
    with st.sidebar:
        # SELECTBOX CHỌN NGÔN NGỮ
        lang_choice = st.selectbox(
            "🌐 " + T("lang_select"),
            ["Tiếng Việt", "English", "中文"],
            index=0
        )
        if lang_choice == "Tiếng Việt": st.session_state.lang = 'vi'
        elif lang_choice == "English": st.session_state.lang = 'en'
        elif lang_choice == "中文": st.session_state.lang = 'zh'
        
        st.divider()
        role_display = T("role_admin") if st.session_state.get("is_admin") else T("role_user")
        st.success(f"👤 {T('welcome')}, {st.session_state.current_user_name} ({role_display})")
        if st.button(T("logout")):
            st.session_state.user_logged_in = False; st.rerun()

    st.title(T("title"))
    
    # TABS (Dùng biến T để dịch)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([T("tab1"), T("tab2"), T("tab3"), T("tab4"), T("tab5")])

    # TAB 1: RAG
    with tab1:
        st.header(T("t1_header"))
        with st.container():
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: file_excel = st.file_uploader(T("t1_up_excel"), type="xlsx", key="t1")
            with c2: uploaded_files = st.file_uploader(T("t1_up_doc"), type=["pdf", "docx", "txt", "md", "html"], accept_multiple_files=True)
            with c3: st.write(""); st.write(""); btn_run = st.button(T("t1_btn"), type="primary", use_container_width=True)

        if btn_run and uploaded_files:
            vec = load_models()
            db, df = None, None
            has_db = False
            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                    db = vec.encode([f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}" for _, r in df.iterrows()])
                    has_db = True
                    st.success(T("t1_connect_ok").format(n=len(df)))
                except: st.error("Error Reading Excel.")

            for f in uploaded_files:
                text = doc_file(f)
                link = ""
                if has_db:
                    q = vec.encode([text[:2000]])
                    sc = cosine_similarity(q, db)[0]
                    idx = np.argsort(sc)[::-1][:3]
                    for i in idx:
                        if sc[i] > 0.35: link += f"- {df.iloc[i]['Tên sách']} ({sc[i]*100:.0f}%)\n"

                with st.spinner(T("t1_analyzing").format(name=f.name)):
                    prompt = f"Analyze '{f.name}'. User Language: {st.session_state.lang}. Related: {link}. Content: {text[:20000]}"
                    # Sử dụng hàm an toàn
                    res = run_gemini_safe(model.generate_content, prompt)
                    if res:
                        st.markdown(f"### 📄 {f.name}"); st.markdown(res.text); st.markdown("---")
                        luu_lich_su_vinh_vien("Phân Tích Sách", f.name, res.text)

        # Graph
        if file_excel:
            try:
                if "df_viz" not in st.session_state: st.session_state.df_viz = pd.read_excel(file_excel).dropna(subset=["Tên sách"])
                df_v = st.session_state.df_viz
                
                with st.expander(T("t1_graph_title"), expanded=False):
                    vec = load_models()
                    if "book_embs" not in st.session_state:
                        with st.spinner("Đang số hóa sách..."):
                            st.session_state.book_embs = vec.encode(df_v["Tên sách"].tolist())
                    
                    embs = st.session_state.book_embs
                    sim = cosine_similarity(embs)
                    nodes, edges = [], []
                    
                    # Graph Config
                    total_books = len(df_v)
                    c_slider1, c_slider2 = st.columns(2)
                    with c_slider1: max_nodes = st.slider("Số lượng sách hiển thị:", 5, total_books, min(50, total_books))
                    with c_slider2: threshold = st.slider("Độ tương đồng nối dây:", 0.0, 1.0, 0.45)

                    for i in range(max_nodes):
                        nodes.append(Node(id=str(i), label=df_v.iloc[i]["Tên sách"], size=20, color="#FFD166"))
                        for j in range(i+1, max_nodes):
                            if sim[i,j]>threshold: edges.append(Edge(source=str(i), target=str(j), color="#118AB2"))
                    
                    config = Config(width=900, height=600, directed=False, physics=True, collapsible=False)
                    agraph(nodes, edges, config)
            except: pass

    # TAB 2: DỊCH
    with tab2:
        st.header(T("t2_header"))
        
        # 1. Input tràn màn hình
        txt = st.text_area(T("t2_input"), height=150, placeholder="Dán văn bản vào đây (Anh/Việt/Trung)...")
        
        # 2. Các nút chọn nằm trên 1 hàng
        c_lang, c_style, c_btn = st.columns([1, 1, 1])
        with c_lang:
            target_lang = st.selectbox(T("t2_target"), ["Tiếng Việt", "English", "中文 (Chinese)", "French", "Japanese"])
        with c_style:
            style = st.selectbox(T("t2_style"), T("t2_styles"))
        with c_btn: 
            st.write(""); st.write("")
            btn_trans = st.button(T("t2_btn"), type="primary", use_container_width=True)

        # 3. Xử lý & Hiển thị kết quả (Tràn màn hình)
        if btn_trans and txt:
            with st.spinner("AI đang xử lý..."):
                prompt = f"""
                Bạn là Chuyên gia Ngôn ngữ.
                Nhiệm vụ: Dịch và phân tích văn bản sau.
                
                YÊU CẦU:
                1. Ngôn ngữ đích: {target_lang}.
                2. Phong cách: {style}.
                3. QUAN TRỌNG: Nếu dịch sang TIẾNG TRUNG, bắt buộc cung cấp: Chữ Hán, Pinyin (có dấu).
                4. Phân tích 3 từ vựng/cấu trúc hay nhất.
                
                Văn bản gốc: "{txt}"
                """
                res = run_gemini_safe(model.generate_content, prompt)
                
                if res:
                    st.markdown("---")
                    st.markdown(res.text)
                    
                    # Nút tải HTML
                    html_content = f"<html><body><h2>Translation</h2><p><b>Original:</b> {txt}</p><hr>{markdown.markdown(res.text)}</body></html>"
                    st.download_button("💾 Download HTML", html_content, "translation.html", "text/html")
                    
                    luu_lich_su_vinh_vien("Dịch Thuật", f"{target_lang}: {txt[:20]}...", res.text)

  # === TAB 3: ĐẤU TRƯỜNG TƯ DUY (MULTI-AGENT ARENA) ===
    with tab3:
        st.header(T("t3_header"))
        
        # 1. CHỌN CHẾ ĐỘ CHƠI
        # Dùng key='mode_select_tab3' để tránh trùng ID với nơi khác
        mode = st.radio("Chọn chế độ:", ["👤 Solo (User vs AI)", "⚔️ Debate (AI vs AI)"], horizontal=True, key="mode_select_tab3")

        # 1.1. DANH SÁCH NHÂN VẬT 
        personas = {
            "😈 Kẻ Phản Biện": "Tìm lỗ hổng logic để tấn công. Phải tìm ra điểm yếu.",
            "🤔 Socrates": "Chỉ đặt câu hỏi (Socratic method). Không đưa ra câu trả lời.",
            "📈 Nhà Kinh Tế Học": "Phân tích mọi vấn đề qua Chi phí, Lợi nhuận (ROI), Cung cầu.",
            "🚀 Steve Jobs": "Đòi hỏi Sự Đột Phá, Tối giản và Trải nghiệm người dùng.",
            "❤️ Người Tri Kỷ": "Lắng nghe, đồng cảm và khích lệ.",
            "⚖️ Immanuel Kant": "Triết gia Lý tính. Đề cao Đạo đức nghĩa vụ, logic chặt chẽ, khô khan.",
            "🔥 Nietzsche": "Triết gia Sinh mệnh. Phá vỡ quy tắc, cổ vũ cho Ý chí quyền lực.",
            "🙏 Phật Tổ": "Góc nhìn Vô ngã, Duyên khởi, Vô thường. Giúp giải cấu trúc sự chấp trước."
        }
        
        st.divider()

        # --- CHẾ ĐỘ 1: SOLO (user vs AI) ---
        if mode == "👤 Solo (User vs AI)":
            # Dùng Container để cô lập không gian ID
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1: 
                    p_sel = st.selectbox(T("t3_persona_label"), list(personas.keys()), key="solo_persona_select")
                with c2: 
                    st.write(""); st.write("")
                    if st.button(T("t3_clear"), key="btn_clr_solo"): 
                        st.session_state.chat_history = []
                        st.rerun()

                # Hiển thị lịch sử chat
                for m in st.session_state.chat_history:
                    st.chat_message(m["role"]).markdown(m["content"])
                
                # Input Chat (Key duy nhất)
                if q := st.chat_input(T("t3_input"), key="chat_input_solo"):
                    st.chat_message("user").markdown(q)
                    st.session_state.chat_history.append({"role":"user", "content":q})
                    
                    # Logic gọi AI
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
                    prompt = f"""
                    VAI TRÒ CỦA BẠN: {personas[p_sel]}
                    LỊCH SỬ CHAT: {history_text}
                    NGƯỜI DÙNG NÓI: "{q}"
                    YÊU CẦU: Phân tích sâu, phản biện sắc sảo, và trả lời bằng ngôn ngữ của người dùng.
                    """
                    
                    res = run_gemini_safe(model.generate_content, prompt)
                    if res:
                        st.chat_message("assistant").markdown(res.text)
                        st.session_state.chat_history.append({"role":"assistant", "content":res.text})
                        luu_lich_su_vinh_vien("Tranh Biện Solo", f"Vs {p_sel}: {q}", res.text)

        # --- CHẾ ĐỘ 2: DEBATE (AI vs AI) ---
        else:
            with st.container():
                st.info("💡 Hướng dẫn: Chọn 2-3 triết gia, đặt chủ đề và xem họ 'đấu võ mồm'.")
                
                participants = st.multiselect("Chọn các Đấu Thủ (Tối đa 3):", list(personas.keys()), default=["⚖️ Immanuel Kant", "🔥 Nietzsche"], key="multi_select_battle")
                topic = st.text_input("Chủ đề Tranh Luận:", placeholder="Ví dụ: Tiền có mua được hạnh phúc không?", key="topic_input_battle")
                
                if "battle_logs" not in st.session_state: st.session_state.battle_logs = []

                col_start, col_clear = st.columns([1, 5])
                with col_start:
                    start_battle = st.button("🔥 KHAI CHIẾN", type="primary", key="btn_start_battle", disabled=(len(participants) < 2))
                with col_clear:
                    if st.button("🗑️ Xóa Bàn", key="btn_clr_battle"):
                        st.session_state.battle_logs = []; st.rerun()

                # Logic chạy vòng lặp tranh luận (3 Vòng)
                if start_battle and topic and len(participants) > 1:
                    st.session_state.battle_logs = []
                    st.session_state.battle_logs.append(f"**📢 CHỦ TỌA:** Khai mạc tranh luận về: *'{topic}'*")
                    
                    with st.status("Hội đồng đang tranh luận nảy lửa (3 vòng)...") as status:
                        for round_num in range(1, 4):
                            status.update(label=f"🔄 Vòng {round_num}/3 đang diễn ra...")
                            
                            for i, p_name in enumerate(participants):
                                if round_num == 1:
                                    p_prompt = f"Bạn là {p_name}. Chủ đề: {topic}. Đưa ra quan điểm đầu tiên."
                                else:
                                    target_index = (i - 1 + len(participants)) % len(participants)
                                    target_name = participants[target_index]
                                    last_speech = ""
                                    for log in reversed(st.session_state.battle_logs):
                                        if log.startswith(f"**{target_name}:**"):
                                            last_speech = log.replace(f"**{target_name}:** ", "")
                                            break
                                    p_prompt = f"VAI TRÒ: {p_name}. PHẢN BÁC: \"{target_name}\" vừa nói: \"{last_speech}\". Yêu cầu: Phản bác lại lập luận đó theo triết lý của bạn."
                                
                                # SỬ DỤNG HÀM AN TOÀN + SLEEP NHIỀU HƠN
                                res = run_gemini_safe(model.generate_content, p_prompt)
                                if res:
                                    reply = res.text
                                    st.session_state.battle_logs.append(f"**{p_name}:** {reply}")
                                    time.sleep(4) # Tăng lên 4 giây để tránh lỗi ResourceExhausted

                        status.update(label="✅ Tranh luận kết thúc! (Đã chạy 3 vòng)", state="complete")
                        
                        full_log = "\n\n".join(st.session_state.battle_logs)
                        luu_lich_su_vinh_vien("Hội Đồng Tranh Biện", topic, full_log)
                        st.toast("💾 Đã lưu biên bản cuộc họp vào Nhật Ký!", icon="✅")
                        
                # Hiển thị kết quả trận đấu
                for log in st.session_state.battle_logs:
                    st.markdown(log)
                    st.markdown("---")

    # TAB 4: TTS (ĐÃ CÓ LẠI GIỌNG NỮ)
    with tab4:
        st.header(T("t4_header"))
        v_opt = {
            "🇻🇳 VN - Nam (Nam Minh)": "vi-VN-NamMinhNeural", 
            "🇻🇳 VN - Nữ (Hoài My)": "vi-VN-HoaiMyNeural",
            "🇺🇸 US - Nam (Andrew)": "en-US-AndrewMultilingualNeural",
            "🇺🇸 US - Nữ (Emma)": "en-US-EmmaNeural",
            "🇨🇳 CN - Nam (Yunjian)": "zh-CN-YunjianNeural",
            "🇨🇳 CN - Nữ (Xiaoyi)": "zh-CN-XiaoyiNeural"
        }
        c1, c2 = st.columns([3,1])
        with c2: 
            v_sel = st.selectbox(T("t4_voice"), list(v_opt.keys()))
            rate = st.slider(T("t4_speed"), -50, 50, 0)
        with c1: inp = st.text_area("Text:", height=200)
        
        if st.button(T("t4_btn"), type="primary", use_container_width=True) and inp:
            try:
                generate_edge_audio_sync(inp, v_opt[v_sel], f"{'+' if rate>=0 else ''}{rate}%", "out.mp3")
                st.audio("out.mp3")
                with open("out.mp3", "rb") as f:
                    st.download_button(T("t4_dl"), f, "audio.mp3", "audio/mpeg")
                luu_lich_su_vinh_vien("Tạo Audio", v_sel, inp)
            except Exception as e: st.error(f"Error: {e}")

    # TAB 5: LỊCH SỬ
    with tab5:
        st.header(T("t5_header"))
        if st.button(T("t5_refresh")):
            st.session_state.history = tai_lich_su_tu_sheet(); st.rerun()
        
        if st.session_state.history:
            try:
                df_h = pd.DataFrame(st.session_state.history)
                df_h["score"] = pd.to_numeric(df_h["sentiment_score"], errors='coerce')
                if not df_h.dropna(subset=["score"]).empty:
                    st.subheader(T("t5_chart"))
                    fig = px.line(df_h, x="time", y="score", color="sentiment_label", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
            except: pass

            for item in reversed(st.session_state.history):
                user_tag = f"👤 [{item.get('user')}] " if st.session_state.is_admin else ""
                with st.expander(f"⏰ {item['time']} | {user_tag}{item['type']} | {item['title']}"):
                    st.markdown(item['content'])
        else:
            st.info(T("t5_empty"))

# --- 6. MAIN ---
def main():
    # Khởi tạo ngôn ngữ mặc định nếu chưa có
    if 'lang' not in st.session_state:
        st.session_state.lang = 'vi'

    pm = PasswordManager()
    if not st.session_state.get("user_logged_in"):
        st.title(T("login_title"))
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            p = st.text_input(T("pass_placeholder"), type="password")
            if st.button(T("login_btn"), use_container_width=True):
                if pm.check_password(p):
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = p
                    st.session_state.current_user_name = st.session_state.key_name_mapping.get(p, "User")
                    
                    # QUAN TRỌNG: Kiểm tra lại quyền Admin tại đây
                    st.session_state.is_admin = pm.is_admin(p)
                    
                    st.rerun()
                else: st.error(T("wrong_pass"))
    else:
        show_main_app()

if __name__ == "__main__":
    main()
