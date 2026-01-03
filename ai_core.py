import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import time

# Exceptions chuẩn
from google.api_core.exceptions import ResourceExhausted as GeminiResourceExhausted
from google.api_core.exceptions import ServiceUnavailable as GeminiServiceUnavailable, InternalServerError as GeminiInternalError
from openai import RateLimitError, APIError, AuthenticationError

class AI_Core:
    def __init__(self):
        self.status_container = st.container()
        self.status_message = st.empty()  # Status động cho từng request
        self.grok_ready = False
        self.gemini_ready = False
        self.deepseek_ready = False
        self.grok_client = None
        self.deepseek_client = None

        # 1. GROK (xAI) - Ưu tiên cao nhất
        try:
            if "xai" in st.secrets and "api_key" in st.secrets["xai"]:
                self.grok_client = OpenAI(
                    api_key=st.secrets["xai"]["api_key"],
                    base_url="https://api.x.ai/v1"
                )
                self.grok_ready = True
        except Exception:
            pass

        # 2. DEEPSEEK
        try:
            if "deepseek" in st.secrets and "api_key" in st.secrets["deepseek"]:
                self.deepseek_client = OpenAI(
                    api_key=st.secrets["deepseek"]["api_key"],
                    base_url="https://api.deepseek.com/v1"
                )
                self.deepseek_ready = True
        except Exception:
            pass
        
        
        # 3. GEMINI
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
                    temperature=0.8,
                    max_output_tokens=7000,
                    top_p=0.95,
                    top_k=40
                )
                self.gemini_ready = True
        except Exception:
            pass

        
        # Status tổng quan
        with self.status_container:
            status_parts = []
            if self.grok_ready: status_parts.append("🟢 Grok")
            if self.gemini_ready: status_parts.append("🟡 Gemini")
            if self.deepseek_ready: status_parts.append("🟣 DeepSeek")
            if not status_parts:
                st.error("🔴 Không có API nào sẵn sàng")
            else:
                st.caption(f"**AI Engine:** {' → '.join(status_parts)}")

    def _grok_generate(self, prompt, system_instruction=None):
        if not self.grok_ready: return None
        models = ["grok-4", "grok-beta", "grok-2"]  # Model thực tế xAI 2026
        messages = [{"role": "user", "content": prompt}]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})

        for model in models:
            try:
                resp = self.grok_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=7000,
                    top_p=0.95
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIError, AuthenticationError):
                time.sleep(5)
                continue
            except Exception:
                continue
        return None

    def _gemini_generate(self, prompt, model_type="flash", system_instruction=None):
        if not self.gemini_ready: return None
        valid_models = {
            "flash": "gemini-2.5-flash",
            "pro": "gemini-2.5-pro"  
        }
        model_name = valid_models.get(model_type, "gemini-2.5-flash")

        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=self.safety_settings,
                generation_config=self.gen_config,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return None
        except (GeminiResourceExhausted, GeminiServiceUnavailable, GeminiInternalError):
            return None
        except Exception:
            return None

    def _deepseek_generate(self, prompt, system_instruction=None):
        if not self.deepseek_ready: return None
        models = ["deepseek-chat", "deepseek-reasoner"]
        messages = [{"role": "user", "content": prompt}]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})

        for model in models:
            try:
                resp = self.deepseek_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=7000
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIError, AuthenticationError):
                time.sleep(5)
                continue
            except Exception:
                continue
        return None

    def generate(self, prompt, model_type="pro", system_instruction=None):
        """GROK → GEMINI → DEEPSEEK - Auto fallback"""
        self.status_message.info("🤖 Đang gọi AI...")

        # 1. GROK
        if self.grok_ready:
            result = self._grok_generate(prompt, system_instruction)
            if result:
                self.status_message.success("🎯 Grok hoàn thành")
                return result

        # 2. DEEPSEEK FREE
        if self.deepseek_ready:
            result = self._deepseek_generate(prompt, system_instruction)
            if result:
                self.status_message.success("💰 DeepSeek hoàn thành")
                return result
        
        
        # 3. GEMINI
        if self.gemini_ready:
            result = self._gemini_generate(prompt, model_type, system_instruction)
            if result:
                self.status_message.success("🔄 Gemini hoàn thành")
                return result

        
        self.status_message.error("⚠️ Tất cả API bận")
        return "⚠️ Hệ thống bận. Thử lại sau 1-2 phút nhé chị!"

    @staticmethod
    @st.cache_data(ttl=3600)
    def analyze_static(text, instruction):
        """RAG dùng DeepSeek (context dài + tiết kiệm)"""
        try:
            if "deepseek" not in st.secrets:
                return "❌ Cần DeepSeek key cho RAG"
            client = OpenAI(
                api_key=st.secrets["deepseek"]["api_key"],
                base_url="https://api.deepseek.com/v1"
            )
            text = text[:180000]  # DeepSeek chịu context dài
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
        except Exception as e:
            return f"❌ RAG lỗi: {str(e)[:150]}"
