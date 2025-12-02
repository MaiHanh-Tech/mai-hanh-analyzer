import logging
import os
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import google.generativeai as genai
import edge_tts
from langdetect import detect

# --- CẤU HÌNH ---
# Lấy Key từ hệ thống (sẽ cấu hình trên Web sau)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Cấu hình Log để theo dõi bot sống hay chết
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- KẾT NỐI NÃO BỘ ---
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')

# Lưu lịch sử chat tạm thời trong RAM
chat_history = {}

# --- CẤU HÌNH GIỌNG ĐỌC ---
VOICE_MAPPING = {
    'vi': 'vi-VN-NamMinhNeural',       
    'en': 'en-US-ChristopherNeural',   
    'zh-cn': 'zh-CN-YunxiNeural',      
    'default': 'vi-VN-NamMinhNeural'   
}

# --- HÀM XỬ LÝ ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Chào Chị Hạnh! Em là VietMaiAI2.0 đây ạ. Em đã lên mây rồi!")

async def chat_with_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id
    
    # Quản lý lịch sử chat
    if chat_id not in chat_history:
        chat_history[chat_id] = model.start_chat(history=[])
    chat = chat_history[chat_id]

    print(f"📩 Nhận tin nhắn: {user_text}") 
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        # 1. Hỏi Gemini
        response = chat.send_message(user_text)
        ai_reply = response.text
        
        # Gửi Text
        await update.message.reply_text(f"📝 {ai_reply}")
        
        # 2. Tạo Giọng nói (Nếu văn bản không quá dài)
        if len(ai_reply) < 1000:
            await context.bot.send_chat_action(chat_id=chat_id, action='record_audio')
            
            # Chọn giọng
            try:
                lang = detect(ai_reply)
            except: lang = 'vi'
            
            voice = VOICE_MAPPING.get(lang.split('-')[0], VOICE_MAPPING['default'])
            if lang == 'zh-cn' or lang == 'zh': voice = VOICE_MAPPING['zh-cn']

            # Tạo file audio
            audio_file = f"voice_{chat_id}.mp3"
            communicate = edge_tts.Communicate(ai_reply, voice)
            await communicate.save(audio_file)
            
            # Gửi Voice
            await update.message.reply_voice(voice=open(audio_file, "rb"))
            os.remove(audio_file) # Dọn dẹp
            
    except Exception as e:
        print(f"Lỗi: {e}")
        await update.message.reply_text("Em đang bị lag một chút, chị hỏi lại nhé!")

# --- CHẠY BOT ---
if __name__ == '__main__':
    if not TELEGRAM_TOKEN or not GOOGLE_API_KEY:
        print("❌ LỖI: Chưa có Token/Key. Hãy cấu hình Environment Variables.")
    else:
        print("🚀 VietMaiAI đang khởi động...")
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_with_ai))
        application.run_polling()
