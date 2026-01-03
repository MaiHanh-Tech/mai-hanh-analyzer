import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import time

# Exceptions
from google.api_core.exceptions import ResourceExhausted as GeminiResourceExhausted
from google.api_core.exceptions import ServiceUnavailable as GeminiServiceUnavailable, InternalServerError as GeminiInternalError
from openai import RateLimitError, APIError, AuthenticationError, Timeout

class AI_Core:
    def __init__(self):
        self.status_container = st.container()
        self.status_message = st.empty()
        self.grok_ready = False
        self.gemini_ready = False
        self.deepseek_ready = False
        self.grok_client = None
        self.deepseek_client = None

        # ✅ TIMEOUT MẶC ĐỊNH
        self.DEFAULT_TIMEOUT = 30  # 30 giây max
        
        # 1. DEEPSEEK
        try:
            if "deepseek" in st.secrets and "api_key" in st.secrets["deepseek"]:
                self.deepseek_client = OpenAI(
                    api_key=st.secrets["deepseek"]["api_key"],
                    base_url="https://api.deepseek.com/v1",
                    timeout=self.DEFAULT_TIMEOUT  # ✅ THÊM TIMEOUT
                )
                self.deepseek_ready = True
        except Exception:
            pass
        
        # 2. GEMINI
        try:
            if "api_keys" in st.secrets and "gemini_api_key" in st.secrets["api_keys"]:
                genai.configure(api_key=st.secrets["api_keys"]["gemini_api_key"])
                self.safety_settings = [
                    {"category": c, "threshold": "BLOCK_NONE"} for c in [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT"
                    ]
                ]
                self.gen_config = genai.GenerationConfig(
                    temperature=0.7,  # ✅ GIẢM từ 0.8 → 0.7 (ít random hơn)
                    max_output_tokens=2000,  # ✅ GIẢM từ 7000 → 2000 (tranh biện ngắn gọn)
                    top_p=0.9,  # ✅ GIẢM từ 0.95 → 0.9
                    top_k=40
                )
                self.gemini_ready = True
        except Exception:
            pass

        # 3. GROK
        try:
            if "xai" in st.secrets and "api_key" in st.secrets["xai"]:
                self.grok_client = OpenAI(
                    api_key=st.secrets["xai"]["api_key"],
                    base_url="https://api.x.ai/v1",
                    timeout=self.DEFAULT_TIMEOUT  # ✅ THÊM TIMEOUT
                )
                self.grok_ready = True
        except Exception:
            pass

        # Status
        with self.status_container:
            status_parts = []
            if self.deepseek_ready: status_parts.append("🟣 DeepSeek")
            if self.gemini_ready: status_parts.append("🟡 Gemini")
            if self.grok_ready: status_parts.append("🟢 Grok")
            if not status_parts:
                st.error("🔴 Không có API nào sẵn sàng")
            else:
                st.caption(f"**AI Engine:** {' → '.join(status_parts)}")

    def _deepseek_generate(self, prompt, system_instruction=None, max_tokens=2000):
        """✅ SỬA: Thêm timeout, giảm max_tokens, bỏ sleep dài"""
        if not self.deepseek_ready: 
            return None
        
        models = ["deepseek-chat"]  # ✅ BỎ reasoner (chậm + đắt)
        messages = [{"role": "user", "content": prompt}]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})

        for model in models:
            try:
                resp = self.deepseek_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,  # ✅ GIẢM
                    max_tokens=max_tokens,  # ✅ ĐỘNG
                    timeout=self.DEFAULT_TIMEOUT  # ✅ THÊM
                )
                return resp.choices[0].message.content.strip()
            except Timeout:
                # ✅ Timeout → Bỏ qua model này
                continue
            except (RateLimitError, APIError):
                time.sleep(2)  # ✅ GIẢM từ 5s → 2s
                continue
            except Exception:
                continue
        return None

    def _gemini_generate(self, prompt, model_type="flash", system_instruction=None):
        """✅ GIỮ NGUYÊN nhưng thêm timeout logic"""
        if not self.gemini_ready: 
            return None
        
        valid_models = {
            "flash": "gemini-2.0-flash-exp",
            "pro": "gemini-2.0-flash-exp"  # ✅ Dùng flash cho cả 2 (nhanh hơn)
        }
        model_name = valid_models.get(model_type, "gemini-2.0-flash-exp")

        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=self.safety_settings,
                generation_config=self.gen_config,
                system_instruction=system_instruction
            )
            # ✅ THÊM: Gemini không có timeout param, dùng try-except
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return None
        except (GeminiResourceExhausted, GeminiServiceUnavailable, GeminiInternalError):
            return None
        except Exception:
            return None

    def _grok_generate(self, prompt, system_instruction=None, max_tokens=2000):
        """✅ SỬA: Thêm timeout, giảm max_tokens"""
        if not self.grok_ready: 
            return None
        
        models = ["grok-beta"]  # ✅ Chỉ dùng 1 model (nhanh hơn)
        messages = [{"role": "user", "content": prompt}]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})

        for model in models:
            try:
                resp = self.grok_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    timeout=self.DEFAULT_TIMEOUT  # ✅ THÊM
                )
                return resp.choices[0].message.content.strip()
            except Timeout:
                continue
            except (RateLimitError, APIError):
                time.sleep(2)  # ✅ GIẢM từ 5s → 2s
                continue
            except Exception:
                continue
        return None

    def generate(self, prompt, model_type="pro", system_instruction=None, max_tokens=2000):
        """
        ✅ CHIẾN LƯỢC MỚI: Gemini FIRST (nhanh nhất)
        Gemini → DeepSeek → Grok
        """
        self.status_message.info("🤖 Đang gọi AI...")

        # ✅ 1. GEMINI FIRST (Nhanh nhất)
        if self.gemini_ready:
            result = self._gemini_generate(prompt, model_type, system_instruction)
            if result:
                self.status_message.success("⚡ Gemini")
                return result
        
        # ✅ 2. DEEPSEEK (Nếu Gemini fail)
        if self.deepseek_ready:
            result = self._deepseek_generate(prompt, system_instruction, max_tokens)
            if result:
                self.status_message.success("💰 DeepSeek")
                return result

        # ✅ 3. GROK (Cuối cùng)
        if self.grok_ready:
            result = self._grok_generate(prompt, system_instruction, max_tokens)
            if result:
                self.status_message.success("🎯 Grok")
                return result
                
        self.status_message.error("⚠️ Tất cả API bận")
        return "⚠️ Hệ thống bận. Thử lại sau!"

    @staticmethod
    @st.cache_data(ttl=3600)
    def analyze_static(text, instruction):
        """✅ RAG dùng Gemini (có cache, nhanh)"""
        try:
            # ✅ Ưu tiên Gemini cho RAG (có cache)
            if "api_keys" in st.secrets and "gemini_api_key" in st.secrets["api_keys"]:
                genai.configure(api_key=st.secrets["api_keys"]["gemini_api_key"])
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                text = text[:150000]  # ✅ Gemini chịu context dài
                response = model.generate_content(f"{instruction}\n\n{text}")
                if response and response.text:
                    return response.text.strip()
            
            # Fallback DeepSeek
            if "deepseek" in st.secrets:
                client = OpenAI(
                    api_key=st.secrets["deepseek"]["api_key"],
                    base_url="https://api.deepseek.com/v1",
                    timeout=60
                )
                text = text[:180000]
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=4000,
                    temperature=0.3
                )
                return resp.choices[0].message.content.strip()
                
            return "❌ Không có API khả dụng"
        except Exception as e:
            return f"❌ RAG lỗi: {str(e)[:150]}"
