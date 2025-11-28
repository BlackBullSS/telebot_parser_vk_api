import requests
import time
import logging
from datetime import datetime, timedelta


def get_wall_posts(group_id, vk_token, weeks=1):
    """
    Получает посты со стены *одной* группы за последние N недель.
    """
    now = int(datetime.now().timestamp())
    week_ago = int((datetime.now() - timedelta(weeks=weeks)).timestamp())

    params = {
        "owner_id": group_id,
        "extended": 0,
        "count": 100,  # Максимум за один вызов
        "access_token": vk_token,
        "v": "5.131",
    }

    try:
        response = requests.get(
            "https://api.vk.com/method/wall.get", params=params, timeout=10
        )
    except requests.exceptions.RequestException as e:
        logging.error(
            f"Ошибка соединения при запросе к API ВК для группы {group_id}: {e}"
        )
        # Возвращаем None или [], чтобы main_loop мог обработать это
        return []
    except Exception as e:
        logging.error(
            f"Неожиданная ошибка при запросе к API ВК для группы {group_id}: {e}"
        )
        return []

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:  # json.decoder.JSONDecodeError в Python 3.5+
            logging.error(
                f"Ответ API ВК для группы {group_id} не в формате JSON: {response.text[:200]}..."
            )
            return []

        if "response" in data and "items" in data["response"]:
            posts = data["response"]["items"]
            # Фильтруем по дате (за последнюю неделю)
            recent_posts = [p for p in posts if p["date"] >= week_ago]
            logging.info(
                f"  - Получено {len(recent_posts)} постов за неделю из группы {group_id}"
            )
            return recent_posts
        else:
            logging.warning(
                f"  - Внимание: Нет 'response' или 'items' в ответе для группы {group_id}: {data}"
            )
    else:
        logging.error(
            f"  - Ошибка HTTP при запросе к API для группы {group_id}: {response.status_code}, {response.text[:200]}..."
        )
    return []


def extract_text_from_post(post):
    """
    Извлекает текст из поста, включая вложения (фото, ссылки и т.д.).
    """
    try:
        text = post.get("text", "")
        attachments = post.get("attachments", [])

        for att in attachments:
            if att["type"] == "photo":
                # Можно добавить ссылку на фото, если нужно
                pass
            elif att["type"] == "link":
                link = att["link"].get("url", "")
                text += f"\nСсылка: {link}"
            # Добавьте другие типы вложений по необходимости

        return text
    except Exception as e:
        logging.error(
            f"Ошибка при извлечении текста из поста {post.get('id', 'unknown')}: {e}"
        )
        # Возвращаем пустую строку, если не удалось извлечь
        return ""
