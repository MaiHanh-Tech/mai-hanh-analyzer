import logging
import os
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import google.generativeai as genai
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading # Để chạy Web và Bot song song

# --- CẤU HÌNH ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- KẾT NỐI BỘ NÃO ---
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_flash = genai.GenerativeModel('gemini-2.5-pro')
    model_pro = genai.GenerativeModel('gemini-2.5-pro')
else:
    print("⚠️ CHÚ Ý: Chưa thấy GOOGLE_API_KEY!")

chat_history[chat_id] = model_flash.start_chat(history=[
    {"role": "user", "parts": "Bạn là một chuyên gia phân tích dữ liệu, trả lời ngắn gọn, khách quan, không dùng từ ngữ cảm xúc."},
    {"role": "model", "parts": "Dạ, em chào Chị Hạnh ạ!"}
])

# --- [MỚI] 1. HÀM XỬ LÝ WEB (ĐÁNH LỪA RENDER) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    """Xử lý yêu cầu HTTP đơn giản (Đánh lừa Render)"""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

def run_web_server():
    """Chạy Server Web giả để giữ Bot sống"""
    # Render yêu cầu dùng cổng PORT lấy từ Environment Variable
    PORT = int(os.environ.get("PORT", 8080)) 
    server = HTTPServer(('', PORT), HealthCheckHandler)
    print(f"🌐 Web Server giả chạy trên cổng {PORT} (Giữ Bot sống)...")
    server.serve_forever()

# --- 2. HÀM XỬ LÝ BOT TELEGRAM (GIỮ NGUYÊN LOGIC CŨ) ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 VietMaiAI đã sống lại! (Hybrid Mode)")

async def chat_with_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    # Logic Gemini (Flash/Pro) giữ nguyên như cũ
    # ... (Chèn logic Gemini Dual-Core vào đây) ...
    try:
        if user_text.lower().startswith("/g "):
            # Logic PRO
            real_prompt = user_text[3:].strip()
            response = model_pro.generate_content(real_prompt)
            ai_reply = f"🧠 **[PRO]**\n{response.text}"
        else:
            # Logic FLASH
            if chat_id not in chat_history:
                chat_history[chat_id] = model_flash.start_chat(history=[])
            response = chat_history[chat_id].send_message(user_text)
            ai_reply = response.text
            
        # Gửi Text
        await update.message.reply_text(ai_reply)
        
    except Exception as e:
        await update.message.reply_text(f"⚠️ Lỗi xử lý: {str(e)}")

# --- 3. HÀM CHẠY CHÍNH (KẾT HỢP CẢ 2) ---
if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        print("❌ LỖI: Chưa có TELEGRAM_TOKEN!")
    else:
        # A. CHẠY WEB SERVER (THREAD RIÊNG)
        web_thread = threading.Thread(target=run_web_server)
        web_thread.start()
        
        # B. CHẠY TELEGRAM BOT (THREAD CHÍNH)
        print("🤖 Bắt đầu Polling Telegram...")
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_with_ai))
        application.run_polling()
