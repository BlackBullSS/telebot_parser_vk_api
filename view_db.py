# view_db.py

import sqlite3
import pandas as pd

DB_NAME = "parsed_posts.db"

def view_processed_posts():
    """
    Подключается к базе данных и выводит таблицу с:
    - raw_posts.text (оригинальный текст объявления)
    - raw_posts.text_translit (транслитерированный текст объявления)
    - processed_posts.processed_text (оригинальный ответ DeepSeek)
    - processed_posts.processed_text_translit (транслитерированный ответ DeepSeek)
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # SQL-запрос для объединения таблиц
    query = """
    SELECT
        rp.text AS original_text,
        rp.text_translit AS translit_text,
        pp.processed_text AS deepseek_original,
        pp.processed_text_translit AS deepseek_translit
    FROM
        raw_posts rp
    JOIN
        processed_posts pp ON rp.id = pp.raw_post_id;
    """

    try:
        # Выполняем запрос и создаем DataFrame
        df = pd.read_sql_query(query, conn)

        # Закрываем соединение
        conn.close()

        # Выводим DataFrame
        print(df)

    except Exception as e:
        print(f"Ошибка при выполнении запроса к базе данных: {e}")
        conn.close()

if __name__ == "__main__":
    view_processed_posts()