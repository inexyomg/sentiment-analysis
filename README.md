# Анализ эмоциональной тональности русскоязычных текстов

Полный исследовательский пайплайн для многоклассовой классификации эмоций в русскоязычных текстах. Разработан в рамках цифровых гуманитарных исследований.

**Задача:** определить одну из 7 базовых эмоций по таксономии Экмана в произвольном русском тексте.

---

## Таксономия эмоций (Ekman, 7 классов)

| ID | Эмоция | Примеры текстов |
|---|---|---|
| 0 | anger / гнев | «Это возмутительно!», «Терпеть невозможно» |
| 1 | disgust / отвращение | «Это омерзительно», «Меня тошнит от этого» |
| 2 | fear / страх | «Мне очень страшно», «Боюсь последствий» |
| 3 | joy / радость | «Я так рад!», «Лучший день в жизни» |
| 4 | sadness / грусть | «Сердце разрывается», «Невозможно смириться» |
| 5 | surprise / удивление | «Не могу поверить!», «Совершенно неожиданно» |
| 6 | neutral / нейтральное | «Совещание перенесли на пятницу» |

---

## Полный пайплайн

```
00_dataset_collection   → Сбор и объединение 8+ датасетов
01_eda_and_preprocessing → EDA, очистка, токенизация
          ↓
    СТРАТЕГИЯ ОБУЧЕНИЯ: двухэтапная
          ↓
07_two_stage_training   → Stage 1: pretrain на большом корпусе (~200k)
                        → Stage 2: fine-tune на чистом нативном RU (~16k)
          ↓
02_train_rubert         → ruBERT-base (одиночное обучение, альтернатива)
03_train_xlmroberta     → XLM-RoBERTa-base
04_train_rubert_tiny    → ruBERT-tiny2
          ↓
05_ensemble_and_results → Hard/Soft voting, Weighted avg, Stacking,
                          Temperature scaling, итоговое сравнение
          ↓
06_dh_applications      → Прикладной анализ для DH-исследований
```

---

## Датасеты

### Эмоциональные (прямая разметка Ekman)

| Датасет | Размер | Язык | Классов | Роль |
|---|---|---|---|---|
| `seara/ru_go_emotions` | ~54k | перевод RU | 7 (из 28) | Stage 1 + 2 |
| `cedr` | ~9.4k | нативный RU | 5 | **Stage 2 (чистый)** |
| `Djacon/ru-izard-emotions` | ~30k | перевод RU | 7 (из 10) | Stage 1 |
| `brighter-dataset/BRIGHTER-emotion-categories` | ~2-3k RU | нативный RU | 6 | **Stage 2 (чистый)** |
| `Aniemore/resd` | ~4.5k | нативный RU | 7 (включая disgust) | **Stage 2 (чистый)** |
| `KELONMYOSA/dusha_emotion_audio` | ~300k (50k cap) | нативный RU | 4 | Stage 1 |

### Сентиментальные (приблизительный маппинг pos→joy, neg→sadness)

| Датасет | Размер | Классов | Роль |
|---|---|---|---|
| `sismetanin/rureviews` | ~90k | 3 | Stage 1 (объём) |
| `sismetanin/rusentitweet` | ~13k | 3 | Stage 1 (объём) |

> **Примечание:** сентиментальные датасеты используются только в Stage 1 для предобучения. В Stage 2 используются исключительно нативные RU датасеты с точной разметкой Ekman.

### Совместное покрытие классов (Stage 2)

```
CEDR:         anger  ·  ·  fear  joy  sadness  surprise  ·
BRIGHTER:     anger  disgust  fear  joy  sadness  surprise  ·
Aniemore:     anger  disgust  fear  joy  sadness  ·  neutral
──────────────────────────────────────────────────────────────
Итого:        anger  disgust  fear  joy  sadness  surprise  neutral ✓ (все 7)
```

---

## Архитектура моделей

| Модель | HuggingFace ID | Параметры | Особенности |
|---|---|---|---|
| **ruBERT** | `blanchefort/rubert-base-cased-sentiment` | ~180M | дообучен на RU сентименте |
| **XLM-RoBERTa** | `xlm-roberta-base` | ~278M | 100 языков, сильный кросс-лингвальный перенос |
| **ruBERT-tiny2** | `cointegrated/rubert-tiny2` | ~12M | быстрый, ~85% качества от base |

---

## Стратегия обучения: двухэтапная (Domain Adaptation)

```
┌────────────────────────────────────────────────────────────────────┐
│  ЭТАП 1: PRETRAIN  (~200k примеров, все источники)                 │
│                                                                    │
│  ru_go_emotions + cedr + ru_izard + dusha + brighter_hf +         │
│  aniemore + rureviews + rusentitweet                               │
│                                                                    │
│  loss = Focal (γ=2.0) + class_weights   lr = 2e-5   epochs = 3   │
│  → Модель учит язык эмоций, справляется с дисбалансом             │
└───────────────────────────┬────────────────────────────────────────┘
                            │  инициализация весов
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│  ЭТАП 2: FINE-TUNE  (~16k примеров, нативный RU)                  │
│                                                                    │
│  cedr + brighter_hf + aniemore/resd                               │
│  (высококачественная нативная разметка, все 7 классов)            │
│                                                                    │
│  loss = CE + label_smoothing=0.05   lr = 5e-6   epochs = 3        │
│  → Модель выравнивается на точную таксономию                       │
└────────────────────────────────────────────────────────────────────┘
```

**Почему это работает:** Stage 1 позволяет модели адаптироваться к эмоциональной лексике на большом объёме, Stage 2 исправляет шум от приблизительного маппинга (negative → sadness) и выравнивает веса под точные Ekman-метки.

---

## Ансамблирование

После обучения трёх моделей применяются следующие методы комбинирования:

| Метод | Описание |
|---|---|
| **Hard Voting** | Большинство голосов по классу |
| **Soft Voting** | Среднее вероятностей |
| **Weighted Averaging** | Взвешивание по F1-macro каждой модели |
| **Stacking (LogReg)** | Мета-модель на вероятностях (без утечки данных) |
| **Stacking (SVM)** | Мета-модель SVM |
| **Temperature Scaling** | Калибровка уверенности через NLL-минимизацию |

Стекинг использует `val_probs.npy` (out-of-fold предсказания) — без утечки тестовых данных.

---

## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Инференс на готовых моделях

```python
from src.inference import EmotionClassifier

clf = EmotionClassifier([
    "results/models/rubert",
    "results/models/xlmroberta",
    "results/models/rubert_tiny",
])

clf.predict("Мне очень страшно идти туда одному")
# [{'fear': 0.71, 'sadness': 0.12, 'anger': 0.06, ...}]

clf.predict_label("Я так рад! Это лучший день!")
# ['joy']
```

### Gradio-демо

```bash
python app/app.py --model_dirs results/models/rubert results/models/xlmroberta
```

### Полный пайплайн (Kaggle T4 GPU)

```
1. Запустить 07_two_stage_training.ipynb  — обучение ансамбля
2. Запустить 05_ensemble_and_results.ipynb — ансамблирование и оценка
3. (опционально) 06_dh_applications.ipynb — прикладной анализ
```

---

## Структура проекта

```
sentiment-analysis/
├── src/
│   ├── data_loader.py    — загрузчики 8+ датасетов, merge_datasets, load_stage2_clean
│   ├── preprocessor.py   — clean_text, lemmatize, preprocess_batch
│   ├── trainer.py        — WeightedTrainer (focal/CE/BCE), train_model, train_two_stage
│   ├── ensemble.py       — soft_voting_proba, stacking, fit_temperature, temperature_scaling
│   ├── evaluation.py     — evaluate_predictions, plot_confusion_matrix, compare_models
│   └── inference.py      — EmotionClassifier (batch inference, ensemble)
│
├── notebooks/
│   ├── 00_dataset_collection.ipynb    — сбор, балансировка, сохранение датасетов
│   ├── 01_eda_and_preprocessing.ipynb — EDA, распределения, примеры текстов
│   ├── 02_train_rubert.ipynb          — одиночное обучение ruBERT
│   ├── 03_train_xlmroberta.ipynb      — одиночное обучение XLM-RoBERTa
│   ├── 04_train_rubert_tiny.ipynb     — одиночное обучение ruBERT-tiny2
│   ├── 05_ensemble_and_results.ipynb  — ансамблирование, температурная калибровка
│   ├── 06_dh_applications.ipynb       — эмоц. дуги, временные ряды, хитмапы, атрибуция
│   └── 07_two_stage_training.ipynb    — двухэтапное обучение ансамбля (рекомендуется)
│
├── app/
│   └── app.py            — Gradio-демо, совместим с HuggingFace Spaces
│
├── configs/
│   └── config.yaml       — конфигурация моделей и обучения
│
├── data/                 — директория для локальных датасетов
├── models/               — директория для чекпоинтов (gitignored)
├── results/              — результаты экспериментов
└── requirements.txt
```

---

## Технические детали

### Работа с дисбалансом классов

- **Focal Loss** (γ=2.0): снижает вес лёгких примеров (joy, neutral), усиливает акцент на редких (disgust, fear)
- **Class Weights**: `sklearn.compute_class_weight("balanced")` — автоматически из распределения
- **MAX_PER_CLASS**: ограничение примеров сверху (по умолчанию 15k) для undersampling

### Предотвращение утечки данных в стекинге

Мета-модель обучается на `val_probs.npy` — предсказаниях базовых моделей на **валидационной** выборке, которую они не видели при обучении.

### Инференс

`EmotionClassifier` поддерживает:
- Произвольное число моделей (ансамбль через `soft_voting_proba`)
- Батчевую обработку (`batch_size=32`)
- Предварительную очистку текста (`clean=True`)
- `predict()`, `predict_proba()`, `predict_label()`

---

## Прикладные инструменты (DH)

Ноутбук `06_dh_applications.ipynb` содержит готовые функции:

- **`emotion_arc(text)`** — эмоциональная дуга нарратива (по предложениям через `razdel`)
- **`emotion_timeline(df, date_col)`** — временной ряд эмоций в датированном корпусе
- **`emotion_heatmap(df, group_col)`** — профили по авторам / жанрам / источникам
- **`radar_chart(profiles)`** — паукообразное сравнение профилей
- **`explain_prediction(text)`** — атрибуция токенов (Integrated Gradients, `transformers-interpret`)
- **`emotion_wordclouds(df)`** — облака слов по эмоциям

---

## Зависимости

```
torch, transformers, datasets, accelerate  — обучение и инференс
scikit-learn, scipy                        — ансамблирование, метрики
pandas, numpy, matplotlib, seaborn         — анализ и визуализация
gradio                                     — веб-демо
razdel                                     — русская сегментация предложений
peft                                       — LoRA (опционально, для больших моделей)
transformers-interpret                     — объяснимость модели
pymorphy2, nltk                            — предобработка текста
```

---

## Ссылки

- [GoEmotions (Google, 2020)](https://arxiv.org/abs/2005.00547)
- [CEDR — Corpus for Emotion Detection in Russian](https://github.com/sag111/Corpus-WS-SentSA)
- [BRIGHTER (SemEval-2025 Task 11)](https://arxiv.org/abs/2502.11926)
- [Dusha — Russian Speech Emotion Dataset (Interspeech 2023)](https://www.isca-archive.org/interspeech_2023/kondratenko23_interspeech.html)
- [Aniemore — Russian Emotional AI](https://huggingface.co/Aniemore)
- [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [Temperature Scaling (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)
