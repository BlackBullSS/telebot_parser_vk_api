# ml_processor.py

import pandas as pd
import numpy as np
import logging
import os
import torch
from torch.utils.data import Dataset

# Попробуем импортировать трансформеры и т.д. сразу, чтобы понять, где проблема
try:
    from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        Trainer,
        TrainingArguments,
    )

    logging.info("MLProcessor: Успешно импортированы компоненты transformers.")
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.error(f"MLProcessor: Ошибка импорта transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

# Импортируем sacrebleu для оценки
try:
    from sacrebleu import sentence_bleu

    BLEU_AVAILABLE = True
    logging.info("MLProcessor: Успешно импортирован sacrebleu.")
except ImportError as e:
    logging.error(
        f"MLProcessor: Ошибка импорта sacrebleu: {e}. Оценка BLEU будет отключена."
    )
    BLEU_AVAILABLE = False

MODEL_DIR = "models/t5_custom"


class TextDataset(Dataset):
    def __init__(self, raw_texts, processed_texts, tokenizer, max_length=512):
        self.raw_texts = raw_texts
        self.processed_texts = processed_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, idx):
        raw_text = str(self.raw_texts[idx])
        processed_text = str(self.processed_texts[idx])

        # Добавляем префикс задачи
        input_text = f"parse_post: {raw_text}"

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                processed_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        labels.input_ids[labels.input_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels.input_ids

        # Убираем лишнюю размерность (batch_size=1)
        for key in model_inputs:
            model_inputs[key] = model_inputs[key].squeeze()

        return model_inputs


class MLProcessor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"MLProcessor: Используемое устройство: {self.device}")
        else:
            self.device = torch.device(
                "cpu"
            )  # Если трансформеры не доступны, используем CPU
            logging.warning(
                "MLProcessor: transformers не доступны, GPU не будет использоваться."
            )
        self.load_model()

    def load_model(self):
        """
        Пытается загрузить обученную модель и токенизатор.
        """
        if not TRANSFORMERS_AVAILABLE:
            logging.error(
                "MLProcessor: transformers не установлены. Загрузка модели невозможна."
            )
            return

        if os.path.exists(MODEL_DIR):
            try:
                # legacy не изменяем (по умолчанию теперь False, но если модель обучалась с True, то так и останется)
                self.tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
                self.model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(
                    self.device
                )
                self.is_trained = True
                logging.info("Обученная модель T5 и токенизатор загружены из файлов.")
            except Exception as e:
                logging.error(f"Ошибка при загрузке модели: {e}")
        else:
            logging.info("Файлы модели не найдены. Будет создана новая модель.")

    def save_model(self):
        """
        Сохраняет обученную модель и токенизатор.
        """
        if not TRANSFORMERS_AVAILABLE:
            logging.error(
                "MLProcessor: transformers не установлены. Сохранение модели невозможно."
            )
            return

        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.model and self.tokenizer:
            try:
                self.model.save_pretrained(MODEL_DIR)
                self.tokenizer.save_pretrained(MODEL_DIR)
                logging.info("Модель T5 и токенизатор сохранены в файлы.")
            except Exception as e:
                logging.error(f"Ошибка при сохранении модели: {e}")

    def prepare_data(self):
        """
        Подготавливает данные для обучения из базы.
        """
        logging.info("Загрузка данных для обучения из базы...")
        # Импортируем здесь, чтобы не вызвать ошибку, если db не нужен без трансформеров
        from db import get_all_processed_pairs

        pairs = get_all_processed_pairs()  # Получаем транслитерированные пары
        if not pairs:
            logging.warning("Нет данных для обучения в базе.")
            return None, None

        raw_texts, processed_texts = zip(*pairs)
        return list(raw_texts), list(processed_texts)

    def train(self):
        """
        Обучает модель на данных из базы.
        """
        if not TRANSFORMERS_AVAILABLE:
            logging.error("transformers или torch не установлены. Обучение невозможно.")
            return False  # Возвращаем False, чтобы main_loop мог обработать

        raw_texts, processed_texts = self.prepare_data()
        if raw_texts is None:
            return False

        logging.info("Начинаем обучение модели T5 для генерации текста...")
        try:
            # Инициализируем модель и токенизатор (или загружаем предыдущую версию для дообучения)
            if not self.model or not self.tokenizer:
                # legacy не изменяем (по умолчанию теперь False, но если модель обучалась с True, то так и останется)
                self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self.model = T5ForConditionalGeneration.from_pretrained("t5-small").to(
                    self.device
                )
                logging.info("Загружена предобученная модель t5-small.")

            dataset = TextDataset(raw_texts, processed_texts, self.tokenizer)

            training_args = TrainingArguments(
                output_dir=MODEL_DIR,
                num_train_epochs=3,
                per_device_train_batch_size=2,  # Уменьшите, если получите OutOfMemory на GPU
                save_steps=10_000,
                logging_steps=1000,
                learning_rate=5e-5,
                save_total_limit=2,
                prediction_loss_only=True,
                # fp16=True, # Включить mixed precision для ускорения и экономии памяти (если GPU поддерживает)
                # report_to=None, # Отключить логирование в wandb/mlflow
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
            )

            trainer.train()

            self.save_model()
            self.is_trained = True
            logging.info("Модель T5 обучена и сохранена.")
            return True

        except Exception as e:  # Ловим любую ошибку, не только ImportError
            logging.error(f"Ошибка при обучении модели: {e}")
            # Выводим тип ошибки для ясности
            logging.error(f"Тип ошибки: {type(e).__name__}")
            return False

    def predict(self, raw_text):
        """
        Предсказывает обработанный текст для сырого текста.
        """
        if not TRANSFORMERS_AVAILABLE:
            logging.error(
                "transformers или torch не установлены. Предсказание невозможно."
            )
            return "Ошибка: ML библиотеки не установлены."

        if not self.is_trained:
            logging.warning("Модель не обучена. Возвращаем пустой ответ.")
            return "Модель не обучена."

        try:
            # Если модель не загружена в память, но файлы есть
            if not self.model or not self.tokenizer:
                # legacy не изменяем (по умолчанию теперь False, но если модель обучалась с True, то так и останется)
                self.tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
                self.model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(
                    self.device
                )
                self.is_trained = True

            input_text = f"parse_post: {raw_text}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,  # Для детерминированного ответа
                )

            predicted_text = self.tokenizer.decode(
                outputs[0].cpu(), skip_special_tokens=True
            )
            logging.debug(f"ML модель предсказала: {predicted_text}")
            return predicted_text

        except Exception as e:
            logging.error(f"Ошибка при предсказании модели: {e}")
            logging.error(f"Тип ошибки: {type(e).__name__}")
            return "Ошибка предсказания модели."

    def evaluate(self):
        """
        Оценивает точность модели, сравнивая её ответы с DeepSeek.
        Использует BLEU score.
        """
        if not BLEU_AVAILABLE:
            logging.warning("sacrebleu не установлен. Оценка BLEU невозможна.")
            return 0.0

        logging.info("Оценка точности модели (BLEU)...")
        # Импортируем функцию из db
        from db import get_all_processed_pairs, get_ml_processed_pairs

        deepseek_pairs = get_all_processed_pairs()  # (raw_translit, deepseek_translit)
        if not deepseek_pairs:
            logging.warning("Нет данных для оценки (DeepSeek).")
            return 0.0

        # Получаем пары (raw_translit, ml_translit) для сравнения
        ml_pairs = get_ml_processed_pairs()  # (raw_translit, ml_translit)
        if not ml_pairs:
            logging.warning("Нет данных для оценки (ML).")
            return 0.0

        # --- ИСПОЛЬЗУЕМ PANDAS ДЛЯ СОПОСТАВЛЕНИЯ ---
        import pandas as pd

        deepseek_df = pd.DataFrame(
            deepseek_pairs, columns=["raw_text", "processed_text"]
        )
        ml_df = pd.DataFrame(ml_pairs, columns=["raw_text", "ml_processed_text"])

        # Слияние по raw_text
        comparison_df = pd.merge(
            deepseek_df,
            ml_df,
            on="raw_text",
            how="inner",
            suffixes=("_deepseek", "_ml"),
        )

        if comparison_df.empty:
            logging.warning(
                "Нет общих данных для оценки BLEU между DeepSeek и ML моделью."
            )
            return 0.0

        deepseek_outputs = comparison_df["processed_text"].tolist()
        # Прогоняем каждый raw_text через модель, чтобы получить актуальный вывод для оценки
        # Вместо использования уже сохранённого ml_processed_text, который мог быть с ошибкой
        raw_texts_for_ml = comparison_df["raw_text"].tolist()
        ml_outputs = []
        for raw_text in raw_texts_for_ml:
            try:
                ml_text = self.predict(raw_text)
                # Проверяем, является ли результат строкой
                if not isinstance(ml_text, str):
                    logging.warning(
                        f"predict вернул не строку: {type(ml_text)}, значение: {repr(ml_text)}. Используем пустую строку."
                    )
                    ml_text = ""
                ml_outputs.append(ml_text)
            except Exception as e:
                logging.error(
                    f"Ошибка при предсказании для текста '{raw_text[:50]}...': {e}"
                )
                # Если predict вызвал ошибку, добавляем пустую строку
                ml_outputs.append("")

        # --- /ИСПОЛЬЗУЕМ PANDAS ДЛЯ СОПОСТАВЛЕНИЯ ---

        # Вычисляем BLEU
        bleu_scores = []
        for i, (ml_text, ref_text) in enumerate(zip(ml_outputs, deepseek_outputs)):
            # sentence_bleu ожидает список ссылок (для кейсов с несколькими эталонами),
            # мы передаём один эталон в виде списка
            try:
                # Убедимся, что оба текста - строки
                if not isinstance(ref_text, str):
                    ref_text = str(ref_text) if ref_text is not None else ""
                if not isinstance(ml_text, str):
                    ml_text = str(ml_text) if ml_text is not None else ""

                # --- ДОБАВИМ ОТЛАДКУ ---
                logging.debug(f"BLEU - Пары {i}:")
                logging.debug(
                    f"  - ML Text Type: {type(ml_text)}, Value: {repr(ml_text)}"
                )
                logging.debug(
                    f"  - Ref Text Type: {type(ref_text)}, Value: {repr(ref_text)}"
                )
                # --- /ДОБАВИМ ОТЛАДКУ ---

                score = sentence_bleu([ref_text], ml_text)
                bleu_scores.append(
                    score.score
                )  # score.score - это значение BLEU от 0 до 100
            except Exception as e:
                logging.error(
                    f"Ошибка при вычислении BLEU для пары {i} (ML: {repr(ml_text)}, Ref: {repr(ref_text)}): {e}"
                )
                # Присваиваем 0 за ошибку или некорректное значение
                bleu_scores.append(0.0)

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        logging.info(f"Средний BLEU: {avg_bleu:.2f}")
        # Возвращаем как долю (0.0 - 1.0), чтобы соответствовать ожиданиям
        # Это значение можно сравнивать с порогом 0.85 (85%)
        return avg_bleu / 100.0
