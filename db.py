# db.py

import sqlite3
import logging

DB_NAME = "parsed_posts.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Таблица для "сырых" постов (оригинал и транслит)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_posts (
            id INTEGER PRIMARY KEY,
            post_id TEXT UNIQUE, -- Уникальный ID поста (group_id_post_id)
            group_id INTEGER,
            text TEXT, -- Оригинальный текст
            text_translit TEXT, -- Текст в транслите (unidecode)
            date INTEGER, -- Unix timestamp
            url TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Таблица для "обработанных" постов (ответов DeepSeek) (оригинал и транслит)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_posts (
            id INTEGER PRIMARY KEY,
            raw_post_id INTEGER UNIQUE, -- Внешний ключ на raw_posts.id
            processed_text TEXT, -- Оригинальный текст DeepSeek
            processed_text_translit TEXT -- Текст DeepSeek в транслите (unidecode)
        )
    """
    )

    # Таблица для "обработанных" постов моделью (хранится оригинал и транслит)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_processed_posts (
            id INTEGER PRIMARY KEY,
            raw_post_id INTEGER UNIQUE, -- Внешний ключ на raw_posts.id
            ml_processed_text TEXT, -- Оригинальный текст модели
            ml_processed_text_translit TEXT -- Текст модели в транслите (unidecode)
        )
    """
    )

    conn.commit()
    conn.close()


def log_raw_post(post_id, group_id, text, date, url):
    """
    Логирует "сырой" пост в базу (оригинал и транслит через unidecode).
    Возвращает ID вставленной строки.
    """
    # Импортируем unidecode здесь, чтобы не вызвать ошибку, если db не нужен без неё
    from unidecode import unidecode

    text_translit = unidecode(text)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO raw_posts (post_id, group_id, text, text_translit, date, url) VALUES (?, ?, ?, ?, ?, ?)",
            (post_id, group_id, text, text_translit, date, url),
        )
        # Получаем ID вставленной строки (или уже существующей)
        cursor.execute("SELECT id FROM raw_posts WHERE post_id = ?", (post_id,))
        raw_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()
        return raw_id
    except Exception as e:
        logging.error(f"Ошибка при логировании сырого поста {post_id}: {e}")
        conn.close()
        return None


def log_processed_post(raw_post_id, processed_text):
    """
    Логирует "обработанный" пост (DeepSeek) в базу (оригинал и транслит через unidecode).
    """
    from unidecode import unidecode

    processed_text_translit = unidecode(processed_text)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO processed_posts (raw_post_id, processed_text, processed_text_translit) VALUES (?, ?, ?)",
            (raw_post_id, processed_text, processed_text_translit),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(
            f"Ошибка при логировании обработанного поста для raw_post_id {raw_post_id}: {e}"
        )
        conn.close()


def log_ml_processed_post(raw_post_id, ml_processed_text):
    """
    Логирует "обработанный" пост моделью в базу (оригинал и транслит через unidecode).
    """
    from unidecode import unidecode

    ml_processed_text_translit = unidecode(ml_processed_text)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO ml_processed_posts (raw_post_id, ml_processed_text, ml_processed_text_translit) VALUES (?, ?, ?)",
            (raw_post_id, ml_processed_text, ml_processed_text_translit),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(
            f"Ошибка при логировании ML-обработанного поста для raw_post_id {raw_post_id}: {e}"
        )
        conn.close()


def is_post_processed(post_id):
    """
    Проверяет, был ли пост обработан (отправлен в телеграм или помечен как ошибочный).
    Используем ту же логику: проверяем пост по post_id.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM raw_posts WHERE post_id = ?", (post_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def mark_post_as_processed(post_id):
    """
    Помечает пост как обработанный.
    В текущей логике, это означает, что он был хотя бы раз обработан через DeepSeek.
    Мы можем просто убедиться, что он есть в raw_posts и processed_posts.
    """
    # На практике, если log_raw_post и log_processed_post были вызваны, пост "обработан".
    # Можно добавить отдельное поле, но для простоты, если он есть в processed_posts, считаем обработанным.
    pass


def get_all_processed_pairs():
    """
    Возвращает список пар (raw_text_translit, processed_text_translit) из базы.
    Используется для обучения модели.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # JOIN raw_posts и processed_posts по raw_post_id, используем транслитерированные поля
    cursor.execute(
        """
        SELECT rp.text_translit, pp.processed_text_translit
        FROM raw_posts rp
        JOIN processed_posts pp ON rp.id = pp.raw_post_id
    """
    )
    pairs = cursor.fetchall()
    conn.close()
    return pairs


def get_ml_processed_pairs():
    """
    Возвращает список пар (raw_text_translit, ml_processed_text_translit) из базы.
    Используется для тестирования модели.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # JOIN raw_posts и ml_processed_posts по raw_post_id, используем транслитерированные поля
    cursor.execute(
        """
        SELECT rp.text_translit, mp.ml_processed_text_translit
        FROM raw_posts rp
        JOIN ml_processed_posts mp ON rp.id = mp.raw_post_id
    """
    )
    pairs = cursor.fetchall()
    conn.close()
    return pairs
