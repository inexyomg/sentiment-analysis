"""
Локальный сайт (Gradio) для тестирования модели классификации эмоций.

Модель грузится один раз при старте — источником может быть локальная папка
или ID репозитория на HuggingFace Hub.

Запуск (локальная папка):
    python app/app.py --model_dirs results/models/distilled_xlmr

Запуск (модель с HuggingFace Hub):
    python app/app.py --model_dirs Kirillx/ru-emotion-distilled-xlmr

Ансамбль из нескольких моделей (soft-voting):
    python app/app.py --model_dirs results/models/rubert results/models/xlmroberta

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference import EmotionClassifier

# Модель по умолчанию, если не переданы --model_dirs / MODEL_DIRS.
DEFAULT_MODEL = "results/models/distilled_xlmr"

EKMAN_LABEL_NAMES = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

EMOTION_RU = {
    "anger": "гнев", "disgust": "отвращение", "fear": "страх", "joy": "радость",
    "sadness": "грусть", "surprise": "удивление", "neutral": "нейтрально",
}
EMOTION_COLORS = {
    "anger": "#e74c3c", "disgust": "#8e44ad", "fear": "#2c3e50", "joy": "#f39c12",
    "sadness": "#3498db", "surprise": "#1abc9c", "neutral": "#95a5a6",
}
EMOTION_EMOJI = {
    "anger": "😠", "disgust": "🤢", "fear": "😨", "joy": "😊",
    "sadness": "😢", "surprise": "😮", "neutral": "😐",
}

# Загруженный один раз классификатор и подпись об источнике модели.
_clf: EmotionClassifier | None = None
_model_label: str = ""


def _probs_bar_chart(label_probs: dict[str, float]) -> plt.Figure:
    labels = [l for l in EKMAN_LABEL_NAMES if l in label_probs] or list(label_probs)
    values = [label_probs[l] for l in labels]
    colors = [EMOTION_COLORS.get(l, "#888888") for l in labels]
    ticks  = [f"{EMOTION_EMOJI.get(l, '')} {EMOTION_RU.get(l, l)}" for l in labels]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(ticks, values, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Вероятность")
    ax.set_title("Распределение эмоций", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def predict(text: str) -> tuple:
    text = (text or "").strip()
    if not text:
        return "—", None, "Введите текст для анализа."
    if _clf is None:
        return "Ошибка", None, "Модель не загружена."

    try:
        probs_dict = _clf.predict(text, top_k=None)[0]
        top_label = max(probs_dict, key=probs_dict.get)
        top_prob = probs_dict[top_label]
        emoji = EMOTION_EMOJI.get(top_label, "")
        ru = EMOTION_RU.get(top_label, top_label)
        headline = f"## {emoji} {ru.upper()}\n**{top_prob:.1%}** уверенность"
        fig = _probs_bar_chart(probs_dict)
        detail = "\n".join(
            f"- {EMOTION_EMOJI.get(k, '')} **{EMOTION_RU.get(k, k)}**: {v:.1%}"
            for k, v in sorted(probs_dict.items(), key=lambda x: -x[1])
        )
        return headline, fig, detail
    except Exception as e:  # noqa: BLE001
        return "Ошибка", None, f"Ошибка инференса: {e}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Анализ эмоций в русском тексте", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎭 Анализ эмоциональной тональности\n"
            "Определение эмоций в русскоязычных текстах по таксономии Экмана (7 классов).\n\n"
            f"**Загруженная модель:** `{_model_label}`"
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Введите текст",
                    placeholder="Например: Мне очень страшно идти туда одному...",
                    lines=4,
                )
                submit_btn = gr.Button("Анализировать", variant="primary")
            with gr.Column(scale=1):
                top_emotion = gr.Markdown(label="Основная эмоция")
                detail_md = gr.Markdown(label="Детали")

        chart_out = gr.Plot(label="Распределение вероятностей")

        gr.Examples(
            examples=[
                ["Я так рад видеть тебя снова! Это лучший день в моей жизни!"],
                ["Мне очень страшно идти туда одному."],
                ["Это просто отвратительно, как они поступили."],
                ["Завтра будет встреча в 10 утра."],
                ["Не могу поверить! Это совершенно неожиданно!"],
                ["Я так устал от всего этого, ничего не радует."],
            ],
            inputs=[text_input],
        )

        submit_btn.click(predict, inputs=[text_input],
                         outputs=[top_emotion, chart_out, detail_md])
        text_input.submit(predict, inputs=[text_input],
                          outputs=[top_emotion, chart_out, detail_md])

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
