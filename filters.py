# filters.py

import re

SPECIFIC_MODELS = []
GENERAL_MODELS_BRANDS = []
GAS_KEYWORDS = []


def load_keywords_from_file(filepath):
    """
    Загружает ключевые слова из файла с разделением по секциям.
    Обновляет глобальные переменные SPECIFIC_MODELS, GENERAL_MODELS_BRANDS, GAS_KEYWORDS.
    """
    global SPECIFIC_MODELS, GENERAL_MODELS_BRANDS, GAS_KEYWORDS

    specific_models = []
    general_models_brands = []
    gas_keywords = []

    current_section = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if "Specific Models" in line:
                    current_section = "specific"
                elif "General Models" in line:
                    current_section = "general"
                elif "Gas-related keywords" in line:
                    current_section = "gas"
                continue
            if line and current_section == "specific":
                specific_models.append(line.lower())
            elif line and current_section == "general":
                general_models_brands.append(line.lower())
            elif line and current_section == "gas":
                gas_keywords.append(line.lower())

    # Обновляем глобальные переменные
    SPECIFIC_MODELS = specific_models
    GENERAL_MODELS_BRANDS = general_models_brands
    GAS_KEYWORDS = gas_keywords


def contains_keyword(text):
    """
    Проверяет, соответствует ли текст усложнённым правилам поиска,
    включая перестановки слов для SPECIFIC_MODELS.
    """
    # Используем глобальные переменные, загруженные через load_keywords_from_file
    text_lower = text.lower()
    # Разбиваем текст на слова для проверки SPECIFIC_MODELS
    text_words = set(re.split(r"\W+", text_lower))

    # Проверка на "точные" модели/бренды (с учётом перестановок)
    for keyword_phrase in SPECIFIC_MODELS:
        # Разбиваем фразу на слова
        keyword_words = set(re.split(r"\W+", keyword_phrase.lower()))
        # Проверяем, является ли множество слов из фразы подмножеством слов из текста
        if keyword_words.issubset(text_words):
            return True  # Нашли совпадение с учётом перестановки

    # Проверка на "общие" модели/бренды + газ (без учёта перестановки для простоты)
    found_general = False
    found_gas = False

    for keyword in GENERAL_MODELS_BRANDS:
        if keyword in text_lower:
            found_general = True
            break

    if found_general:
        for keyword in GAS_KEYWORDS:
            if keyword in text_lower:
                found_gas = True
                break

    if found_general and found_gas:
        return True

    return False
