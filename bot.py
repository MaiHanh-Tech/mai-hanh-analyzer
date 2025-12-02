import logging
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import google.generativeai as genai
import edge_tts
from langdetect import detect

# --- CẤU HÌNH ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- KẾT NỐI 2 BỘ NÃO (FLASH & PRO) ---
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # 1. Não Nhanh (Flash) - Dùng để chat thường
    model_flash = genai.GenerativeModel('gemini-2.5-flash')
    
    # 2. Não Khủng (Pro) - Dùng khi gõ /g (Hiện tại Google chưa có 2.5, dùng 1.5 Pro là mạnh nhất)
    # Nếu sau này có 2.5, chị chỉ cần sửa tên ở đây
    model_pro = genai.GenerativeModel('gemini-2.5-pro') 
else:
    print("⚠️ CẢNH BÁO: Chưa thấy GOOGLE_API_KEY!")

# Lưu lịch sử chat (Chỉ dùng cho Flash để tiết kiệm nhớ)
chat_history = {}

# --- CẤU HÌNH GIỌNG ĐỌC ---
VOICE_MAPPING = {
    'vi': 'vi-VN-NamMinhNeural',       
    'en': 'en-US-ChristopherNeural',   
    'zh': 'zh-CN-YunjianNeural',      
    'default': 'vi-VN-NamMinhNeural'   
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
    Chào Chị Hạnh! Em là VietMaiAI2.0 (Dual-Core).
    
    ⚡ **Chế độ thường:** Chat tự nhiên (Dùng Flash - Nhanh).
    🧠 **Chế độ Chuyên gia:** Gõ `/g <câu hỏi>` để phân tích sâu (Dùng Pro).
    Ví dụ: `/g Phân tích tâm lý học trong giấc mơ`
    """
    await update.message.reply_text(msg)

async def chat_with_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not GOOGLE_API_KEY:
        await update.message.reply_text("❌ Lỗi Key Render!")
        return

    user_text = update.message.text
    chat_id = update.effective_chat.id
    
    print(f"📩 Nhận tin: {user_text}") 
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        ai_reply = ""
        is_pro_mode = False

        # --- LOGIC ĐỊNH TUYẾN (ROUTING) ---
        
        # TRƯỜNG HỢP 1: DÙNG PRO (Nếu có lệnh /g)
        if user_text.lower().startswith("/g "):
            is_pro_mode = True
            # Cắt bỏ chữ "/g " ở đầu
            real_prompt = user_text[3:].strip()
            
            await update.message.reply_text("🧠 Đang bật chế độ Chuyên gia (Pro)... Chị đợi chút nhé.")
            await context.bot.send_chat_action(chat_id=chat_id, action='typing')
            
            # Gọi Model Pro (Không dùng lịch sử để tập trung vào câu hỏi này)
            response = model_pro.generate_content(real_prompt)
            ai_reply = f"🦁 **[PRO ANALYSIS]**\n{response.text}"

        # TRƯỜNG HỢP 2: DÙNG FLASH (Chat thường)
        else:
            # Quản lý lịch sử chat cho Flash
            if chat_id not in chat_history:
                chat_history[chat_id] = model_flash.start_chat(history=[
                    {"role": "user", "parts": "Bạn là trợ lý thân thiện, trả lời ngắn gọn, tình cảm."},
                    {"role": "model", "parts": "Dạ, em chào Chị Hạnh ạ!"}
                ])
            chat = chat_history[chat_id]
            
            response = chat.send_message(user_text)
            ai_reply = response.text

        # --- GỬI KẾT QUẢ ---
        
        # 1. Gửi Text
        # Nếu dài quá thì chia nhỏ tin nhắn (Telegram giới hạn 4096 ký tự)
        if len(ai_reply) > 4000:
            for x in range(0, len(ai_reply), 4000):
                await update.message.reply_text(ai_reply[x:x+4000])
        else:
            await update.message.reply_text(ai_reply)
        
        # 2. Tạo Giọng nói (Chỉ tạo nếu văn bản ngắn < 1000 ký tự để đỡ spam voice)
        # Pro thường trả lời rất dài nên ta hạn chế đọc voice của Pro trừ khi ngắn
        if len(ai_reply) < 1000:
            await context.bot.send_chat_action(chat_id=chat_id, action='record_audio')
            
            try:
                # Bỏ cái prefix "[PRO]" ra trước khi đọc cho đỡ kỳ
                text_to_speak = ai_reply.replace("🦁 **[PRO ANALYSIS]**", "").strip()
                
                lang_code = detect(text_to_speak)
            except: lang_code = 'vi'
            
            short_lang = lang_code.split('-')[0]
            voice = VOICE_MAPPING.get(short_lang, VOICE_MAPPING['default'])
            if short_lang == 'zh': voice = VOICE_MAPPING['zh']

            audio_file = f"voice_{chat_id}.mp3"
            communicate = edge_tts.Communicate(text_to_speak, voice)
            await communicate.save(audio_file)
            
            await update.message.reply_voice(voice=open(audio_file, "rb"))
            
            if os.path.exists(audio_file):
                os.remove(audio_file)
            
    except Exception as e:
        print(f"Lỗi: {e}")
        await update.message.reply_text(f"⚠️ Bot gặp chút trục trặc: {str(e)}")

# --- CHẠY BOT ---
if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        print("❌ LỖI: Chưa có TELEGRAM_TOKEN!")
    else:
        print("🚀 VietMaiAI2.0 (Dual-Core) đang khởi động...")
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        # Xử lý mọi tin nhắn văn bản (bao gồm cả /g vì nó là text)
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_with_ai))
        application.run_polling()
