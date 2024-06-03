import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext
from pydub import AudioSegment
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Логгирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
API_TOKEN = os.getenv('TG_API_TOKEN')

whisper_models = {
    # "en": whisper.load_model("medium.en"),
    "ru": whisper.load_model("medium")
}


# пока что будет только русский язык
default_language = "ru"

supported_extensions = ['ogg', 'wav', 'mp3', 'm4a']


def speech_to_text(audio_path, model):
    try:
        result = model.transcribe(audio_path)
        return result['text'].strip()
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
        return f"Произошла ошибка: {e}"

def split_long_message(text, max_length=4050):
    messages = []
    current_message = ""

    while len(text) > max_length:
        last_period_index = text.rfind(".", 0, max_length)
        if last_period_index == -1:
            # Если нет точек в пределах max_length, просто разделите по максимальной длине
            messages.append(text[:max_length])
            text = text[max_length:]
        else:
            messages.append(text[:last_period_index + 1])  # Включаем последнюю точку
            text = text[last_period_index + 1:]

    if text:
        messages.append(text)

    return messages

def summarize_text(text):
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)["input_ids"]
    output_ids = model.generate(input_ids=input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary


# Стартовая команда бла бла бла
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        '🤖 Привет, я AI протоколист! Отправьте мне голосовое сообщение или аудиофайл, и я переведу его в текст и сделаю суммаризацию.')

async def handle_audio(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data
    language = user_data.get('language', default_language)
    model = whisper_models[language]

    # Determine if the message contains a voice message or an audio file
    if update.message.voice:
        file = await context.bot.get_file(update.message.voice.file_id)
        file_extension = 'ogg'
        await update.message.reply_text('🤖 Обработка вашего голосового сообщения...')
    elif update.message.audio:
        file = await context.bot.get_file(update.message.audio.file_id)
        file_extension = update.message.audio.mime_type.split('/')[1]
        await update.message.reply_text('🤖 Обработка вашего аудиофайла...')
    elif update.message.document:
        file = await context.bot.get_file(update.message.document.file_id)
        file_extension = update.message.document.file_name.split('.')[-1]
        await update.message.reply_text('🤖 Обработка вашего аудиофайла...')
    else:
        await update.message.reply_text('❌ Пожалуйста, отправьте голосовое сообщение или аудиофайл.')
        return

    # Create the directory if it does not exist
    os.makedirs('audio', exist_ok=True)

    # Define the path to save the file
    file_path = os.path.join('audio', f'{file.file_id}.{file_extension}')
    await file.download_to_drive(file_path)
    logger.info(f"Файл загружен: {file_path}")

    # Convert to WAV if necessary
    if file_extension != 'wav':
        try:
            audio = AudioSegment.from_file(file_path)
            wav_path = file_path.replace(f'.{file_extension}', '.wav')
            audio.export(wav_path, format='wav')
            logger.info(f"Файл конвертирован в WAV: {wav_path}")
            os.remove(file_path)
            file_path = wav_path
        except Exception as e:
            logger.error(f"Ошибка конвертации аудио: {e}")
            await update.message.reply_text(f"Ошибка конвертации аудио: {e}")
            return

    # Convert speech to text
    text = speech_to_text(file_path, model)
    # Split and send the recognized text if it's too long
    for part in split_long_message(text):
        await update.message.reply_text('🗣 Распознанный текст:\n' + part)
    await update.message.reply_text('🤖 Суммаризируем текст, подождите еще чуть чуть...')

    # Summarize text
    summary = summarize_text(text)
    await update.message.reply_text('🤖 Суммаризация текста:\n' + summary)

    # Clean up
    os.remove(file_path)


# Менять язык
async def language(update: Update, context: CallbackContext) -> None:
    keyboard = [
        [InlineKeyboardButton("English", callback_data='en'), InlineKeyboardButton("Русский", callback_data='ru')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Выберите язык:', reply_markup=reply_markup)


# Кнопка чтобы поменять язык
async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    language = query.data
    context.user_data['language'] = language
    await query.edit_message_text(text=f"🤖 Язык установлен на {'Скоро добавим...' if language == 'en' else 'Русский'}")


def main():
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(API_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("language", language))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.ALL, handle_audio))

    # Start the Bot
    application.run_polling()


if __name__ == '__main__':
    main()
