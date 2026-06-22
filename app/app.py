"""
Локальный сайт (Gradio) для распознавания эмоций в русском тексте.

Простой интерфейс: поле ввода → кнопка «Анализировать» → диаграмма
вероятностей по 7 эмоциям (таксономия Экмана).

Модель грузится один раз при старте; источник — локальная папка или
ID репозитория на HuggingFace Hub.

Запуск (локальная папка):
    python app/app.py --model_dirs results/models/distilled_xlmr

Запуск (модель с HuggingFace Hub):
    python app/app.py --model_dirs Kirillx/ru-emotion-distilled-xlmr

Переменные окружения:
    MODEL_DIRS  — список моделей через запятую (папки или HF Hub ID)
    DEVICE      — 'cuda' | 'cpu' (определяется автоматически, если не задано)
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference import EmotionClassifier

# Модель по умолчанию, если не переданы --model_dirs / MODEL_DIRS.
DEFAULT_MODEL = "results/models/distilled_xlmr"

EKMAN_LABEL_NAMES = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EMOTION_COLORS = {
    "anger": "#e74c3c", "disgust": "#8e44ad", "fear": "#2c3e50", "joy": "#f39c12",
    "sadness": "#3498db", "surprise": "#1abc9c", "neutral": "#95a5a6",
}

# Загруженный один раз классификатор и подпись об источнике модели.
_clf: EmotionClassifier | None = None
_model_label: str = ""


def _probs_bar_chart(label_probs: dict[str, float]) -> plt.Figure:
    labels = [l for l in EKMAN_LABEL_NAMES if l in label_probs] or list(label_probs)
    values = [label_probs[l] for l in labels]
    colors = [EMOTION_COLORS.get(l, "#888888") for l in labels]

    top_label = max(label_probs, key=label_probs.get)
    top_prob = label_probs[top_label]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Вероятность")
    ax.set_title(f"Предсказание: {top_label.upper()} ({top_prob:.1%})", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def predict(text: str):
    text = (text or "").strip()
    if not text:
        return None
    if _clf is None:
        raise gr.Error("Модель не загружена.")
    probs_dict = _clf.predict(text, top_k=None)[0]
    return _probs_bar_chart(probs_dict)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Анализ эмоций в русском тексте", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎭 Анализ эмоций в русском тексте\n"
            f"Модель: `{_model_label}` · 7 эмоций (таксономия Экмана)"
        )
        text_input = gr.Textbox(
            label="Введите текст",
            placeholder="Например: Мне очень страшно идти туда одному...",
            lines=4,
        )
        submit_btn = gr.Button("Анализировать", variant="primary")
        chart_out = gr.Plot(label="Вероятности эмоций")

        gr.Examples(
            examples=[
                ["Я так рад видеть тебя снова! Это лучший день в моей жизни!"],
                ["Мне очень страшно идти туда одному."],
                ["Это просто отвратительно, как они поступили."],
                ["Завтра будет встреча в 10 утра."],
                ["Не могу поверить! Это совершенно неожиданно!"],
            ],
            inputs=[text_input],
        )

        submit_btn.click(predict, inputs=[text_input], outputs=[chart_out])
        text_input.submit(predict, inputs=[text_input], outputs=[chart_out])

    return demo


def _load_classifier(model_dirs: list[str]) -> None:
    global _clf, _model_label
    print(f"Загрузка модели: {model_dirs} ...")
    _clf = EmotionClassifier(model_dirs, clean=True)
    _model_label = ", ".join(model_dirs)
    print(f"Готово. Устройство: {_clf.device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs="+", default=[],
                        help="Папки с моделями или HF Hub ID")
    parser.add_argument("--share", action="store_true", help="Публичная ссылка Gradio")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    env_dirs = os.environ.get("MODEL_DIRS", "")
    model_dirs = (args.model_dirs
                  or [d.strip() for d in env_dirs.split(",") if d.strip()]
                  or [DEFAULT_MODEL])

    _load_classifier(model_dirs)
    demo = build_demo()
    demo.launch(server_port=args.port, share=args.share)
