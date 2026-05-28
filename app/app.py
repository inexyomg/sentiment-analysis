"""
Gradio demo for Russian emotion classification.

Usage (local):
    python app/app.py --model_dirs results/models/rubert results/models/xlmroberta

Usage (HuggingFace Spaces):
    Set MODEL_DIRS env variable to a comma-separated list of HF model IDs or local paths.

Environment variables:
    MODEL_DIRS  — comma-separated model directories / HF Hub IDs
    DEVICE      — 'cuda' | 'cpu' (auto-detected if unset)
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
from src.data_loader import EKMAN_LABEL_NAMES

EMOTION_COLORS = {
    "anger":    "#e74c3c",
    "disgust":  "#8e44ad",
    "fear":     "#2c3e50",
    "joy":      "#f39c12",
    "sadness":  "#3498db",
    "surprise": "#1abc9c",
    "neutral":  "#95a5a6",
}
EMOTION_EMOJI = {
    "anger":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "joy":      "😊",
    "sadness":  "😢",
    "surprise": "😮",
    "neutral":  "😐",
}

_clf: EmotionClassifier | None = None


def _get_classifier(model_dirs: list[str]) -> EmotionClassifier:
    global _clf
    if _clf is None:
        _clf = EmotionClassifier(model_dirs, clean=True)
    return _clf


def _probs_bar_chart(label_probs: dict[str, float]) -> plt.Figure:
    labels = EKMAN_LABEL_NAMES
    values = [label_probs.get(l, 0.0) for l in labels]
    colors = [EMOTION_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Distribution", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def predict(text: str, model_dirs_str: str) -> tuple:
    text = text.strip()
    if not text:
        return "—", None, "Введите текст для анализа."

    model_dirs = [d.strip() for d in model_dirs_str.split(",") if d.strip()]
    if not model_dirs:
        return "—", None, "Укажите путь к модели."

    try:
        clf = _get_classifier(model_dirs)
        probs_dict = clf.predict(text, top_k=None)[0]
        top_label = max(probs_dict, key=probs_dict.get)
        top_prob = probs_dict[top_label]
        emoji = EMOTION_EMOJI.get(top_label, "")
        headline = f"{emoji} **{top_label.upper()}** ({top_prob:.1%})"
        fig = _probs_bar_chart(probs_dict)
        detail_lines = [f"- **{k}**: {v:.1%}" for k, v in sorted(probs_dict.items(), key=lambda x: -x[1])]
        detail = "\n".join(detail_lines)
        return headline, fig, detail
    except Exception as e:
        return "Ошибка", None, f"Ошибка: {e}"


def build_demo(model_dirs: list[str]) -> gr.Blocks:
    default_dirs = ", ".join(model_dirs) if model_dirs else "results/models/rubert"

    with gr.Blocks(title="Анализ эмоций в русском тексте", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Анализ эмоциональной тональности\n"
            "Определение эмоций в русскоязычных текстах по таксономии Экмана (7 классов)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Введите текст",
                    placeholder="Например: Мне очень страшно идти туда одному...",
                    lines=4,
                )
                model_dirs_input = gr.Textbox(
                    label="Пути к моделям (через запятую)",
                    value=default_dirs,
                    lines=1,
                )
                submit_btn = gr.Button("Анализировать", variant="primary")

            with gr.Column(scale=1):
                top_emotion = gr.Markdown(label="Основная эмоция")
                detail_md   = gr.Markdown(label="Детали")

        chart_out = gr.Plot(label="Распределение вероятностей")

        gr.Examples(
            examples=[
                ["Я так рад видеть тебя снова! Это лучший день в моей жизни!", default_dirs],
                ["Мне очень страшно идти туда одному.", default_dirs],
                ["Это просто отвратительно, как они поступили.", default_dirs],
                ["Завтра будет встреча в 10 утра.", default_dirs],
                ["Не могу поверить! Это совершенно неожиданно!", default_dirs],
            ],
            inputs=[text_input, model_dirs_input],
        )

        submit_btn.click(
            fn=predict,
            inputs=[text_input, model_dirs_input],
            outputs=[top_emotion, chart_out, detail_md],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs="+", default=[],
                        help="Paths to trained model directories or HF Hub IDs")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    env_dirs = os.environ.get("MODEL_DIRS", "")
    model_dirs = args.model_dirs or [d.strip() for d in env_dirs.split(",") if d.strip()]

    demo = build_demo(model_dirs)
    demo.launch(server_port=args.port, share=args.share)
