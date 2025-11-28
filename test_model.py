# test_model.py

from ml_processor import MLProcessor

# Инициализируем процессор (он загрузит обученную модель)
processor = MLProcessor()

if processor.is_trained:
    print("Модель загружена.")
    # Примеры текста
    test_texts = [
        "Продаю Tokyo Marui MWS GBBR, состояние отличное, 20000р, Москва.",
        "Продаю AK-74, б/у, 15000р.",
        "Ищу MP7, цена 25000р, срочно!",
        # Добавьте свои примеры
    ]

    for text in test_texts:
        input_text = f"parse_post: {text}"
        print(f"\n--- Вход ---\n{input_text}")
        prediction = processor.predict(text)
        print(f"--- Вывод модели ---\n{prediction}")
else:
    print("Модель не обучена или не может быть загружена.")