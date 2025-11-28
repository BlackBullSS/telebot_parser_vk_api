# transliterator.py

from transliterate import translit, get_available_language_codes
import logging
import re

# Проверяем, доступен ли русский
if "ru" not in get_available_language_codes():
    logging.error("Язык 'ru' не доступен в transliterate. Проверьте установку.")
    raise ImportError("Требуется поддержка русского языка в библиотеке transliterate.")


def preprocess_text(text):
    """
    Предварительная обработка текста: замена \n и других пробельных символов на один пробел,
    удаление эмодзи и прочих специфичных символов, которые могут исказить транслит.
    """
    # Заменяем \n и другие пробельные символы (например, \r, \t) на один пробел
    text = re.sub(r"\s+", " ", text)
    # Удаляем эмодзи (простой способ - удалить символы из определённых диапазонов Unicode)
    # Этот паттерн удаляет большинство эмодзи и других символов в диапазонах, где они обычно находятся
    text = re.sub(
        r"[\U0001F600-\U0001F64F"  # Emoticons
        r"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        r"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        r"\U00002600-\U000027BF"  # Miscellaneous Symbols
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U0001F018-\U0001F270"  # Additional emoticons и др.
        r"]+",
        "",
        text,
    )
    # Удаляем лишние пробелы в начале и конце
    text = text.strip()
    return text


def cyrillic_to_latin(text):
    """
    Предварительно обрабатывает и транслитерирует кириллический текст в латиницу (транслит).
    """
    try:
        processed_text = preprocess_text(text)
        # logging.debug(f"Транслитерация: Обработанный текст: '{processed_text}'") # Для отладки
        return translit(processed_text, "ru", reversed=False)
    except Exception as e:
        logging.warning(
            f"Ошибка транслитерации кириллицы в латиницу для текста '{text[:50]}...': {e}"
        )
        # Возвращаем предварительно обработанный текст, если транслитерация не удалась
        # Хотя это маловероятно, если ошибка не в translit самой по себе
        return preprocess_text(text)


def latin_to_cyrillic(text):
    """
    Транслитерирует латинский текст (транслит) обратно в кириллицу.
    """
    try:
        # translit(text, 'ru', reversed=True) -> латиница -> кириллица
        # logging.debug(f"Обратная транслитерация: Входной текст: '{text}'") # Для отладки
        return translit(text, "ru", reversed=True)
    except Exception as e:
        logging.warning(
            f"Ошибка транслитерации латиницы в кириллицу для текста '{text[:50]}...': {e}"
        )
        # Возвращаем исходный текст, если транслитерация не удалась
        return text
