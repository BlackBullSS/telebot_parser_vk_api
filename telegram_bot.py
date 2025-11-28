import telebot
import os
import logging
from dotenv import load_dotenv

load_dotenv()
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"), parse_mode="Markdown")

# --- ID тем в супергруппе ---
# НУЖНО ЗАМЕНИТЬ НА РЕАЛЬНЫЕ ID ТЕМ!
TOPIC_ID_GBB = 85  # ID темы "Поиск GBB" (например, 101)
TOPIC_ID_BARAKHOLKI = 88  # ID темы "Барахолки" (например, 102)
# --- /ID тем ---


def send_post_to_chat(chat_id, url, processed_text, is_suitable=True):
    """
    Отправляет сообщение в чат в *указанную тему*.
    is_suitable: True -> тема "Поиск GBB", False -> тема "Барахолки"
    """
    message = f"🔗 [Новый пост]({url})\n\n{processed_text}"
    # Выбираем ID темы
    topic_id = TOPIC_ID_GBB if is_suitable else TOPIC_ID_BARAKHOLKI

    try:
        bot.send_message(
            chat_id=chat_id,
            text=message,
            message_thread_id=topic_id,  # Указываем ID темы
        )
        logging.debug(f"Сообщение успешно отправлено в чат {chat_id}, тема {topic_id}")
    except telebot.apihelper.ApiException as e:
        # Конкретная ошибка от Telegram API
        logging.error(
            f"Ошибка API Telegram при отправке в чат {chat_id}, тема {topic_id}: {e}"
        )
        # Не поднимаем исключение, main_loop сам обработает
        raise e
    except Exception as e:
        # Любая другая ошибка (например, network error)
        logging.error(
            f"Неожиданная ошибка при отправке в Telegram в чат {chat_id}, тема {topic_id}: {e}"
        )
        # Не поднимаем исключение, main_loop сам обработает
        raise e
