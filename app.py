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
            for item in data:
                formatted.append({
                    "time": item.get("Time", ""),
                    "type": item.get("Type", ""),
                    "title": item.get("Title", ""),
                    "content": item.get("Content", ""),
                    "sentiment_score": float(item.get("SentimentScore", 0.0)),
                    "sentiment_label": item.get("SentimentLabel", "Neutral"),
                })
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


# --- 4b. HÀM EDGE TTS ---
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
        if cloud_data:
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
    except Exception
