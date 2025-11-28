# test_deepseek_translit.py

import os
import re
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# --- Импорты ---
from deepseek_processor import process_with_deepseek

# --- Импорты для транслитерации ---
# 1. transliterate (ваш текущий метод)
from transliterator import (
    preprocess_text as preprocess_text_default,
    cyrillic_to_latin as translit_to_latin_default,
    latin_to_cyrillic as translit_to_cyrillic_default,
)

# 1.1. unidecode (альтернатива)
try:
    from unidecode import unidecode

    UNIDECODE_AVAILABLE = True
    print("Библиотека unidecode доступна.")
except ImportError:
    UNIDECODE_AVAILABLE = False
    print(
        "Библиотека unidecode НЕ доступна. Установите её с помощью 'pip install unidecode'."
    )

    def unidecode(text):
        return text


# 2. Кастомный словарь
# Словарь транслитерации кириллица -> латиница
CYRILLIC_TO_LATIN_MAP = {
    "А": "A",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Е": "E",
    "Ё": "E",
    "Ж": "Zh",
    "З": "Z",
    "И": "I",
    "Й": "I",
    "К": "K",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ф": "F",
    "Х": "Kh",
    "Ц": "Ts",
    "Ч": "Ch",
    "Ш": "Sh",
    "Щ": "Shch",
    "Ъ": "",
    "Ы": "Y",
    "Ь": "",
    "Э": "E",
    "Ю": "Yu",
    "Я": "Ya",
    # Маленькие буквы
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

# Словарь обратной транслитерации латиница -> кириллица
LATIN_TO_CYRILLIC_MAP = {v: k for k, v in CYRILLIC_TO_LATIN_MAP.items()}


def preprocess_text(text):
    """
    Предварительная обработка текста: замена \n и других пробельных символов на один пробел,
    удаление эмодзи и прочих специфичных символов.
    """
    # Заменяем \n и другие пробельные символы (например, \r, \t) на один пробел
    text = re.sub(r"\s+", " ", text)
    # Удаляем эмодзи
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F270]+",
        "",
        text,
    )
    # Удаляем лишние пробелы в начале и конце
    text = text.strip()
    return text


def custom_cyrillic_to_latin(text):
    """
    Транслитерация кириллицы в латиницу с использованием кастомного словаря.
    Обрабатывает только кириллические буквы, остальные символы остаются как есть.
    """
    processed_text = preprocess_text(text)
    result = ""
    for char in processed_text:
        # Проверяем, есть ли символ в словаре, иначе оставляем как есть
        result += CYRILLIC_TO_LATIN_MAP.get(char, char)
    return result


def custom_latin_to_cyrillic(text):
    """
    Обратная транслитерация латиницы в кириллицу с использованием кастомного словаря.
    Пытается сопоставить *последовательности* латинских букв с кириллическими.
    """
    # Этот метод работает по буквам/последовательностям, которые есть в словаре.
    # Он НЕ будет конвертировать слова типа "AK", "CM", "ASR" и т.д.
    # Он конвертирует только то, что находится в LATIN_TO_CYRILLIC_MAP.
    result = ""
    i = 0
    while i < len(text):
        # Пробуем найти самое длинное совпадение (например, 'Shch' до 'Sh')
        found = False
        for length in range(
            min(6, len(text) - i), 0, -1
        ):  # Макс длина транслита для одной буквы (Shch = 4)
            substr = text[i : i + length]
            if substr in LATIN_TO_CYRILLIC_MAP:
                result += LATIN_TO_CYRILLIC_MAP[substr]
                i += length
                found = True
                break
        if not found:
            result += text[i]
            i += 1
    return result


def main():
    # Запрашиваем текст у пользователя
    input_text = input("Введите текст для отправки в DeepSeek: ")

    print("\n--- 1. Оригинальный текст ---")
    print(input_text)

    # --- 2. Предварительная обработка ---
    processed_text = preprocess_text(input_text)
    print(
        "\n--- 1.1. Предварительно обработанный текст (без эмодзи, \\n -> пробел) ---"
    )
    print(processed_text)

    # --- 3. Транслитерация способ 1: transliterate (ваш текущий метод) ---
    print("\n--- 2.1 Текст в транслите (transliterate) ---")
    text_translit_1 = translit_to_latin_default(
        processed_text
    )  # Используем уже предварительно обработанный текст
    print(text_translit_1)

    # --- 4. Транслитерация способ 2: unidecode ---
    print("\n--- 2.2 Текст в ASCII (unidecode) ---")
    if UNIDECODE_AVAILABLE:
        text_unidecode = unidecode(processed_text)
        print(text_unidecode)
    else:
        print("Библиотека недоступна, пропускаем.")
        text_unidecode = processed_text

    # --- 5. Транслитерация способ 3: кастомный словарь ---
    print("\n--- 2.3 Текст в транслите (кастомный словарь) ---")
    text_translit_3 = custom_cyrillic_to_latin(processed_text)
    print(text_translit_3)

    # --- 6. Отправка запроса в DeepSeek (используем результат transliterate) ---
    print(
        "\n--- 3. Отправляем запрос в DeepSeek (используя транслит transliterate или unidecode)... ---"
    )
    text_to_send = processed_text  # Значение по умолчанию
    if UNIDECODE_AVAILABLE:
        text_to_send = text_unidecode
        print("unidecode доступна, используем её.")
    else:
        text_to_send = text_translit_1
        print("unidecode недоступна, используем transliterate.")

    try:
        processed_text_from_deepseek = process_with_deepseek(text_to_send)
        print("Ответ от DeepSeek (в транслите, как он пришёл):")
        print(processed_text_from_deepseek)
    except Exception as e:
        print(f"Ошибка при запросе к DeepSeek: {e}")
        return

    # --- 7. Обратная транслитерация ответа DeepSeek ---
    print("\n--- 4.1 Обратная транслитерация ответа (transliterate) ---")
    back_to_cyrillic_1 = translit_to_cyrillic_default(processed_text_from_deepseek)
    print(back_to_cyrillic_1)

    # --- НОВЫЙ БЛОК ---
    print(
        "\n--- 4.2 Обратная транслитерация ответа (unidecode - НЕТ, это односторонняя операция) ---"
    )
    print("unidecode не поддерживает обратную транслитерацию.")
    # --- /НОВЫЙ БЛОК ---

    print("\n--- 4.3 Обратная транслитерация ответа (кастомный словарь) ---")
    back_to_cyrillic_3 = custom_latin_to_cyrillic(processed_text_from_deepseek)
    print(back_to_cyrillic_3)


if __name__ == "__main__":
    main()
