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
import plotly.express as px
import markdown
import edge_tts
import asyncio

from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
import json
import re

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Mai Hanh Super App", layout="wide", page_icon="💎")


# --- 2. CLASS QUẢN LÝ MẬT KHẨU ---
class PasswordManager:
    def __init__(self):
        self.user_tiers = st.secrets.get("user_tiers", {})
        if "key_name_mapping" not in st.session_state:
            st.session_state.key_name_mapping = {}

    def check_password(self, password):
        if not password:
            return False

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

        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = (
                creds_dict["private_key"]
                .replace("\\n", "\n")
                .replace("\\n", "\n")
            )

        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict, scope
        )
        client = gspread.authorize(creds)

        return client.open("AI_History_Logs").sheet1
    except Exception:
        return None


def phan_tich_cam_xuc(text: str):
    """
    Trả về (score, label) với score ~ [-1,1]
    """
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        sentiment_model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        Hãy phân tích cảm xúc của đoạn nội dung sau và trả về kết quả ở dạng JSON thuần:
        - sentiment_score: một số trong khoảng [-1, 1] (âm = tiêu cực, dương = tích cực)
        - sentiment_label: một trong các giá trị: "Negative", "Neutral", "Positive"

        Nội dung: \"\"\"{text[:2000]}\"\"\"
        """

        res = sentiment_model.generate_content(prompt)
        raw = res.text or ""

        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return 0.0, "Neutral"
        data = json.loads(m.group(0))

        score = float(data.get("sentiment_score", 0.0))
        label = str(data.get("sentiment_label", "Neutral"))
        return score, label
    except Exception:
        return 0.0, "Neutral"


def luu_lich_su_vinh_vien(loai, tieu_de, noi_dung):
    thoi_gian = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Phân tích cảm xúc
    score, label = phan_tich_cam_xuc(noi_dung)

    # 2. Lưu RAM
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(
        {
            "time": thoi_gian,
            "type": loai,
            "title": tieu_de,
            "content": noi_dung,
            "sentiment_score": score,
            "sentiment_label": label,
        }
    )

    # 3. Lưu Cloud
    try:
        sheet = connect_gsheet()
        if sheet:
            # Append: Time, Type, Title, Content, SentimentScore, SentimentLabel
            sheet.append_row(
                [thoi_gian, loai, tieu_de, noi_dung, score, label]
            )
    except Exception:
        pass


def tai_lich_su_tu_sheet():
    try:
        sheet = connect_gsheet()
        if sheet:
            data = sheet.get_all_records()
            formatted = []
            for item in 
                formatted.append(
                    {
                        "time": item.get("Time", ""),
                        "type": item.get("Type", ""),
                        "title": item.get("Title", ""),
                        "content": item.get("Content", ""),
                        "sentiment_score": item.get("SentimentScore", 0.0),
                        "sentiment_label": item.get(
                            "SentimentLabel", "Neutral"
                        ),
                    }
                )
            return formatted
    except Exception:
        return []
    return []


# --- 4. CÁC HÀM XỬ LÝ AI & FILE ---
@st.cache_resource
def load_models():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def doc_file(uploaded_file):
    if not uploaded_file:
        return ""
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
    except Exception:
        return ""
    return ""


# --- 4b. HÀM EDGE TTS (GIỮ LẠI ĐỂ APP CHẠY ĐƯỢC KHI DỊCH VỤ ỔN) ---
async def _edge_tts_generate(text, voice_code, rate, out_path):
    communicate = edge_tts.Communicate(text, voice_code, rate=rate)
    await communicate.save(out_path)


def generate_edge_audio_sync(text, voice_code, rate, out_path="studio_output.mp3"):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(
                _edge_tts_generate(text, voice_code, rate, out_path)
            )
            new_loop.close()
            asyncio.set_event_loop(loop)
        else:
            loop.run_until_complete(
                _edge_tts_generate(text, voice_code, rate, out_path)
            )
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(
            _edge_tts_generate(text, voice_code, rate, out_path)
        )
        new_loop.close()


# --- 5. GIAO DIỆN CHÍNH ---
def show_main_app():
    # Load history
    if "history_loaded" not in st.session_state:
        cloud_data = tai_lich_su_tu_sheet()
        if cloud_
            st.session_state.history = cloud_data
        st.session_state.history_loaded = True

    if "history" not in st.session_state:
        st.session_state.history = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Cấu hình Gemini
    try:
        sys_api_key = st.secrets["system"]["gemini_api_key"]
        genai.configure(api_key=sys_api_key)
        try:
            model = genai.GenerativeModel("gemini-2.5-pro")
        except Exception:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception:
                model = genai.GenerativeModel("gemini-pro")
    except Exception:
        st.error("❌ Lỗi: Chưa cấu hình [system] gemini_api_key trong Secrets!")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.success(f"👤 User: {st.session_state.current_user_name}")
        if st.button("Đăng Xuất"):
            st.session_state.user_logged_in = False
            st.rerun()

    st.title("💎 The Mai Hanh Super-App")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📚 Phân Tích Sách",
            "✍️ Dịch Giả",
            "🗣️ Tranh Biện",
            "🎙️ Phòng Thu AI",
            "⏳ Lịch Sử",
        ]
    )

    # === TAB 1: PHÂN TÍCH SÁCH ===
    with tab1:
        st.header("Trợ lý Nghiên cứu RAG")

        with st.container():
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                file_excel = st.file_uploader(
                    "1. Kết nối Kho Sách", type="xlsx", key="tab1_excel"
                )
            with c2:
                uploaded_files = st.file_uploader(
                    "2. Tài liệu mới",
                    type=["pdf", "docx", "txt", "md", "html"],
                    accept_multiple_files=True,
                )
            with c3:
                st.write("")
                st.write("")
                btn_run = st.button(
                    "🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True
                )

        st.divider()

        if btn_run and uploaded_files:
            vec_model = load_models()
            db_vec, df = None, None
            has_db = False

            if file_excel:
                try:
                    df = pd.read_excel(file_excel).dropna(
                        subset=["Tên sách"]
                    )
                    content = [
                        f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}"
                        for _, r in df.iterrows()
                    ]
                    db_vec = vec_model.encode(content)
                    has_db = True
                    st.success(f"✅ Đã kết nối {len(df)} cuốn sách.")
                except Exception:
                    st.error("Lỗi đọc Excel.")

            for f in uploaded_files:
                text = doc_file(f)
                lien_ket = ""
                if has_db:
                    q_vec = vec_model.encode([text[:2000]])
                    scores = cosine_similarity(q_vec, db_vec)[0]
                    top = np.argsort(scores)[::-1][:3]
                    for idx in top:
                        if scores[idx] > 0.35:
                            lien_ket += (
                                f"- {df.iloc[idx]['Tên sách']} "
                                f"(Khớp: {scores[idx]*100:.1f}%)\n"
                            )

                with st.spinner(f"Đang phân tích {f.name}..."):
                    prompt = (
                        f"Phân tích tài liệu '{f.name}'. "
                        f"Liên kết cũ: {lien_ket}. "
                        f"Nội dung: {text[:20000]}"
                    )
                    res = model.generate_content(prompt)

                    st.markdown(f"### 📄 Kết quả: {f.name}")
                    st.markdown(res.text)
                    st.markdown("---")
                    luu_lich_su_vinh_vien("Phân Tích", f.name, res.text)

        # Biểu đồ + BẢN ĐỒ TƯ DUY
        if file_excel:
            try:
                if "df_viz" not in st.session_state:
                    st.session_state.df_viz = pd.read_excel(
                        file_excel
                    ).dropna(subset=["Tên sách"])
                df_v = st.session_state.df_viz

                # --- 1) Biểu đồ cũ ---
                with st.expander("📊 Thống Kê Kho Sách", expanded=True):
                    g1, g2 = st.columns(2)
                    with g1:
                        if "Tác giả" in df_v.columns:
                            top_auth = (
                                df_v["Tác giả"]
                                .value_counts()
                                .head(10)
                                .reset_index()
                            )
                            top_auth.columns = ["Tác giả", "Số lượng"]
                            st.plotly_chart(
                                px.bar(
                                    top_auth,
                                    x="Số lượng",
                                    y="Tác giả",
                                    orientation="h",
                                    title="Top Tác giả",
                                ),
                                use_container_width=True,
                            )
                    with g2:
                        if "CẢM NHẬN" in df_v.columns:
                            df_v["Len"] = df_v["CẢM NHẬN"].apply(
                                lambda x: len(str(x))
                            )
                            st.plotly_chart(
                                px.histogram(
                                    df_v,
                                    x="Len",
                                    title="Độ sâu Review",
                                ),
                                use_container_width=True,
                            )

                # --- 2) BẢN ĐỒ TƯ DUY SÁCH ---
                with st.expander(
                    "🪐 Bản đồ tư duy sách (Interactive Knowledge Graph)",
                    expanded=False,
                ):
                    st.caption(
                        "Mỗi cuốn sách là một hành tinh; đường nối dựa trên độ tương đồng nội dung (cosine similarity)."
                    )

                    vec_model = load_models()
                    if "book_embeddings" not in st.session_state:
                        contents = [
                            f"{r['Tên sách']} {str(r.get('CẢM NHẬN',''))}"
                            for _, r in df_v.iterrows()
                        ]
                        st.session_state.book_embeddings = vec_model.encode(
                            contents
                        )
                        st.session_state.book_titles = df_v[
                            "Tên sách"
                        ].tolist()

                    embs = st.session_state.book_embeddings
                    titles = st.session_state.book_titles

                    if len(titles) == 0:
                        st.info("Chưa có dữ liệu sách để vẽ bản đồ.")
                    else:
                        max_default = min(30, len(titles))
                        max_nodes = st.slider(
                            "Số sách tối đa hiển thị:",
                            10,
                            min(100, len(titles)),
                            max_default,
                        )

                        sim_matrix = cosine_similarity(embs, embs)

                        nodes = []
                        edges = []

                        for i, title in enumerate(titles[:max_nodes]):
                            nodes.append(
                                Node(
                                    id=str(i),
                                    label=title,
                                    size=20,
                                    color="#FFD166",
                                )
                            )

                        threshold = st.slider(
                            "Ngưỡng liên kết (cosine similarity):",
                            0.3,
                            0.95,
                            0.6,
                            0.05,
                        )
                        n = min(max_nodes, len(titles))
                        for i in range(n):
                            for j in range(i + 1, n):
                                score = sim_matrix[i, j]
                                if score >= threshold:
                                    edges.append(
                                        Edge(
                                            source=str(i),
                                            target=str(j),
                                            label=f"{score:.2f}",
                                        )
                                    )

                        config = Config(
                            width="100%",
                            height=600,
                            directed=False,
                            physics=True,
                            hierarchical=False,
                            nodeHighlightBehavior=True,
                            highlightColor="#EF476F",
                        )

                        return_value = agraph(
                            nodes=nodes, edges=edges, config=config
                        )

                        if return_value and return_value.get(
                            "selected_node"
                        ):
                            idx = int(return_value["selected_node"])
                            st.markdown(
                                f"### 📘 Đang zoom vào: **{titles[idx]}**"
                            )
                            sims = sim_matrix[idx]
                            top_idx = sims.argsort()[::-1][1:6]
                            rel = [
                                (titles[j], sims[j])
                                for j in top_idx
                                if sims[j] >= threshold
                            ]
                            if rel:
                                st.write("Các hành tinh liên quan:")
                                for name, s in rel:
                                    st.write(
                                        f"- {name} *(Độ liên quan: {s:.2f})*"
                                    )
                            else:
                                st.info(
                                    "Sách này chưa có 'hành tinh' nào đủ gần theo ngưỡng hiện tại."
                                )
            except Exception:
                pass

    # === TAB 2: DỊCH GIẢ ===
    with tab2:
        st.header("Dịch Thuật Đa Chiều")

        txt_in = st.text_area(
            "Nhập văn bản cần dịch:",
            height=150,
            placeholder="Dán tiếng Việt, Anh hoặc Trung vào đây...",
        )

        c_opt, c_btn = st.columns([3, 1])
        with c_opt:
            style_opt = st.selectbox(
                "Chọn Phong Cách Dịch:",
                [
                    "Mặc định (Trung tính)",
                    "Hàn lâm/Học thuật",
                    "Văn học/Cảm xúc",
                    "Đời thường/Dễ hiểu",
                    "Thương mại/Kinh tế",
                    "Kiếm hiệp/Cổ trang",
                ],
            )
        with c_btn:
            st.write("")
            st.write("")
            btn_trans = st.button(
                "✍️ Dịch Ngay", type="primary", use_container_width=True
            )

        if btn_trans and txt_in:
            with st.spinner("AI đang tư duy..."):
                prompt = f"""
                Bạn là Chuyên gia Ngôn ngữ. Hãy xử lý văn bản sau: "{txt_in}"
                
                YÊU CẦU:
                1. Tự động nhận diện ngôn ngữ nguồn.
                2. Nếu là Tiếng Việt -> Dịch sang Tiếng Anh và Tiếng Trung (Kèm Pinyin).
                3. Nếu là Ngoại ngữ -> Dịch sang Tiếng Việt.
                4. PHONG CÁCH DỊCH: {style_opt}.
                5. Phân tích 3 từ vựng/cấu trúc hay nhất.
                
                TRÌNH BÀY: Dùng Markdown rõ ràng.
                """
                res = model.generate_content(prompt)

                st.markdown("### 🎯 Kết Quả:")
                st.markdown(res.text)

                html_content = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: sans-serif; padding: 20px; line-height: 1.6; }}
                    </style>
                </head>
                <body>
                    <h2>Bản Dịch ({style_opt})</h2>
                    <div style="background: #f0f2f6; padding: 15px; border-radius: 5px;">
                        <strong>Gốc:</strong><br>{txt_in}
                    </div>
                    <hr>
                    {markdown.markdown(res.text)}
                </body>
                </html>
                """
                st.download_button(
                    label="💾 Tải kết quả (HTML)",
                    data=html_content,
                    file_name="Ban_Dich.html",
                    mime="text/html",
                )

                luu_lich_su_vinh_vien(
                    "Dịch Thuật", f"{style_opt}: {txt_in[:20]}...", res.text
                )

    # === TAB 3: TRANH BIỆN ===
    with tab3:
        st.header("Luyện Tư Duy")
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).markdown(msg["content"])

        if query := st.chat_input("Chủ đề tranh luận..."):
            st.chat_message("user").markdown(query)
            st.session_state.chat_history.append(
                {"role": "user", "content": query}
            )

            prompt = f"Phản biện lại quan điểm này: '{query}'"
            res = model.generate_content(prompt)

            st.chat_message("assistant").markdown(res.text)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": res.text}
            )

            luu_lich_su_vinh_vien("Tranh Biện", query, res.text)

    # === TAB 4: PHÒNG THU AI (EDGE TTS) ===
    with tab4:
        st.header("🎙️ Phòng Thu AI Đa Ngôn Ngữ")
        st.caption("Công nghệ lõi: Microsoft Edge TTS (có thể không ổn định).")

        voice_options = {
            "🇻🇳 Việt - Nam (Nam Minh - Trầm ấm)": "vi-VN-NamMinhNeural",
            "🇻🇳 Việt - Nữ (Hoài My - Ngọt ngào)": "vi-VN-HoaiMyNeural",
            "🇺🇸 Anh - Nam (Andrew - Trầm, Đa ngôn ngữ)": "en-US-AndrewMultilingualNeural",
            "🇺🇸 Anh - Nữ (Emma - Tự nhiên, Thanh toát)": "en-US-EmmaNeural",
            "🇨🇳 Trung - Nam (Yunjian - Thể thao, Khỏe khoắn)": "zh-CN-YunjianNeural",
            "🇨🇳 Trung - Nữ (Xiaoyi - Nhẹ nhàng, Tình cảm)": "zh-CN-XiaoyiNeural",
        }

        c_text, c_config = st.columns([3, 1])
        with c_config:
            st.markdown("#### 🎛️ Cấu hình")
            selected_label = st.selectbox(
                "Chọn Giọng Đọc:", list(voice_options.keys())
            )
            selected_voice_code = voice_options[selected_label]

            speed = st.slider("Tốc độ:", -50, 50, 0, format="%d%%")
            rate_str = f"{'+' if speed >= 0 else ''}{speed}%"

        with c_text:
            MAX_CHARS = 4000
            input_text = st.text_area(
                "Nhập văn bản:",
                height=250,
                placeholder="Dán nội dung vào đây... (nên là câu hoàn chỉnh, hạn chế chỉ dùng ký tự đặc biệt)",
            )
            char_count = len(input_text)
            st.caption(f"Độ dài: {char_count}/{MAX_CHARS} ký tự")

        if st.button(
            "🔊 BẮT ĐẦU TẠO AUDIO",
            type="primary",
            use_container_width=True,
            disabled=(char_count == 0),
        ):
            if char_count == 0:
                st.error("⚠️ Vui lòng nhập nội dung.")
            elif char_count > MAX_CHARS:
                st.error(
                    f"⚠️ Quá dài! Vui lòng cắt bớt dưới {MAX_CHARS} ký tự."
                )
            elif len("".join(ch for ch in input_text if ch.isalpha())) < 5:
                st.error(
                    "⚠️ Nội dung quá ít chữ cái (chỉ toàn ký tự đặc biệt?). Hãy nhập câu đầy đủ hơn."
                )
            else:
                with st.spinner(
                    "Đang tạo audio từ Microsoft Edge TTS..."
                ):
                    try:
                        out_file = "studio_output.mp3"
                        generate_edge_audio_sync(
                            input_text, selected_voice_code, rate_str, out_file
                        )

                        st.success(
                            f"✅ Đã tạo xong với giọng: {selected_label}"
                        )
                        st.audio(out_file, format="audio/mp3")

                        with open(out_file, "rb") as f:
                            file_bytes = f.read()
                        st.download_button(
                            label="⬇️ TẢI FILE MP3",
                            data=file_bytes,
                            file_name=f"audio_{datetime.now().strftime('%H%M%S')}.mp3",
                            mime="audio/mpeg",
                        )

                        try:
                            luu_lich_su_vinh_vien(
                                "Tạo Audio",
                                selected_label,
                                input_text[:50],
                            )
                        except Exception:
                            pass

                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
                        st.info(
                            "💡 Nếu lỗi 'No audio was received' hoặc 401, đó là do dịch vụ Edge TTS phía Microsoft. "
                            "Hãy thử:\n"
                            "- Rút ngắn nội dung.\n"
                            "- Đổi sang giọng khác.\n"
                            "- Dùng môi trường/ mạng khác.\n"
                            "- Hoặc chuyển sang dịch vụ TTS khác ổn định hơn."
                        )

    # === TAB 5: LỊCH SỬ + NHẬT KÝ CẢM XÚC ===
    with tab5:
        st.header("Kho Lưu Trữ & Nhật ký cảm xúc")

        if st.button("🔄 Tải lại Lịch sử"):
            st.session_state.history = tai_lich_su_tu_sheet()
            st.rerun()

        history = st.session_state.history

        if history:
            # 1) Biểu đồ Mood Timeline
            try:
                df_hist = pd.DataFrame(history)
                df_hist["time_dt"] = pd.to_datetime(
                    df_hist["time"], errors="coerce"
                )
                df_hist = df_hist.dropna(subset=["time_dt"])

                if "sentiment_score" in df_hist.columns:
                    df_hist["sentiment_score"] = pd.to_numeric(
                        df_hist["sentiment_score"], errors="coerce"
                    )
                    df_sent = df_hist.dropna(subset=["sentiment_score"])

                    if not df_sent.empty:
                        st.subheader("📈 Nhật ký cảm xúc theo thời gian")
                        st.caption(
                            "Score ~ [-1,1]: âm = tiêu cực, dương = tích cực."
                        )

                        fig = px.line(
                            df_sent.sort_values("time_dt"),
                            x="time_dt",
                            y="sentiment_score",
                            color="sentiment_label",
                            markers=True,
                            title="Mood Timeline",
                        )
                        fig.update_layout(
                            xaxis_title="Thời gian",
                            yaxis_title="Sentiment Score",
                        )
                        st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning(
                    "Không vẽ được biểu đồ cảm xúc. Kiểm tra lại dữ liệu trong Sheet."
                )

            st.markdown("---")
            st.subheader("📚 Chi tiết từng bản ghi")

            for item in reversed(history):
                senti_txt = ""
                if "sentiment_label" in item:
                    senti_txt = (
                        f" | Mood: {item.get('sentiment_label', 'N/A')} "
                        f"({item.get('sentiment_score', 0)})"
                    )
                with st.expander(
                    f"⏰ {item['time']} | {item['type']} | {item['title']}{senti_txt}"
                ):
                    st.markdown(item["content"])
        else:
            st.info("Chưa có lịch sử.")


# --- 6. MAIN ---
def main():
    pm = PasswordManager()
    if not st.session_state.get("user_logged_in", False):
        st.title("🔐 Mai Hạnh Login")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_pass = st.text_input("Password:", type="password")
            if st.button("Login", use_container_width=True):
                if pm.check_password(user_pass):
                    st.session_state.user_logged_in = True
                    st.session_state.current_user = user_pass
                    st.session_state.current_user_name = (
                        st.session_state.key_name_mapping.get(
                            user_pass, "User"
                        )
                    )
                    st.rerun()
                else:
                    st.error("Sai mật khẩu!")
    else:
        show_main_app()


if __name__ == "__main__":
    main()
