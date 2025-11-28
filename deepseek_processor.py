# deepseek_processor.py

from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",  # Или другой URL, если отличается
)


def process_with_deepseek(text):
    """
    Отправляет текст в DeepSeek и возвращает структурированный ответ.
    """
    # --- Изменённый промт ---
    prompt = f"""
    Проанализируй следующий текст объявления и выдели ключевую информацию в следующем формате:

    1) Бренд, модель:
    2) Цена:
    3) Состояние/Тюнинг/Описание:
    4) Город:

    Текст: {text}
    В ответе должно быть только информация с объявления и ничего больше.
    """
    # --- /Изменённый промт ---

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # или другой подходящий
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1,
        )
        logging.debug("Текст успешно обработан через DeepSeek.")
        return response.choices[0].message.content
    except client._exceptions.APIConnectionError as e:
        logging.error(f"Ошибка подключения к DeepSeek API: {e}")
        # Возвращаем строку-заглушку
        return "Ошибка обработки текста: проблема с подключением к DeepSeek."
    except client._exceptions.AuthenticationError as e:
        logging.error(f"Ошибка аутентификации в DeepSeek API: {e}")
        # Критическая ошибка, но не останавливаем программу
        return "Ошибка обработки текста: ошибка аутентификации в DeepSeek."
    except client._exceptions.RateLimitError as e:
        logging.error(f"Превышено ограничение запросов к DeepSeek API: {e}")
        # Возвращаем строку-заглушку
        return "Ошибка обработки текста: превышено ограничение запросов к DeepSeek."
    except Exception as e:
        logging.error(f"Неожиданная ошибка при запросе к DeepSeek: {e}")
        # Возвращаем строку-заглушку
        return "Ошибка обработки текста."
