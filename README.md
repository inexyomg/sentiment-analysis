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

## Структура пайплайна (4 блока)

```
┌─────────────────────────────────────────────────────────────────┐
│  Блок 1: 01_data_preparation.ipynb                              │
│  Загрузка 8+ датасетов → очистка → аугментация (rut5 + BT)     │
│  Сохранение stage1_data_augmented / stage2_data_augmented       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Блок 2: 02_training.ipynb                                      │
│  Stage 1: pretrain на большом корпусе (focal loss, lr=2e-5)     │
│  Stage 2: fine-tune на чистом нативном RU (CE, lr=5e-6)         │
│  3 модели: ruBERT · XLM-RoBERTa · ruBERT-tiny2                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Блок 3: 03_ensemble.ipynb                                      │
│  Soft voting · Stacking · Temperature scaling                   │
│  Финальная оценка по всем классам                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Блок 4: 04_applications.ipynb                                  │
│  DH-инструменты: эмоц. дуги, временные ряды, хитмапы           │
│  + Gradio-демо (app/app.py)                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Датасеты

### Эмоциональные (прямая разметка Ekman)

| Датасет | Размер | Тип | Классов | Роль |
|---|---|---|---|---|
| `seara/ru_go_emotions` | **~211k** (raw) | перевод RU | 7 из 28 | Stage 1 |
| `Djacon/ru-izard-emotions` | ~30k | перевод RU | 7 из 10 | Stage 1 |
| `cedr` | ~9.4k | **нативный RU** | 5 | Stage 1 + **Stage 2** |
| `brighter-dataset/BRIGHTER-emotion-categories` | ~3k RU | **нативный RU** | 6 | Stage 1 + **Stage 2** |
| `Aniemore/resd` | ~4.5k | **нативный RU** | 7 (incl. disgust) | Stage 1 + **Stage 2** |
| `SberDevices/Dusha` | ~300k | **нативный RU** | 4 | Stage 1 |

### Сентиментальные (приблизительный маппинг)

| Датасет | Размер | Маппинг | Роль |
|---|---|---|---|
| `sismetanin/rureviews` | ~90k | pos→joy, neg→sadness | Stage 1 (объём) |
| `sismetanin/rusentitweet` | ~13k | pos→joy, neg→sadness | Stage 1 (объём) |

### Покрытие редких классов по источникам

| Класс | GoEmotions raw | Izard | CEDR | BRIGHTER | Aniemore | Dusha |
|---|---|---|---|---|---|---|
| anger | ✓ | ✓ | ✓ | ✓ | ✓ | **✓ (основной)** |
| disgust | ✓ | ✓ | ✗ | ✓ | **✓** | ✗ |
| fear | ✓ | ✓ | ✓ | ✓ | **✓** | ✗ |

> Disgust и fear редки в тексте (~0.6% и ~2.8% в GoEmotions). После аугментации каждый класс доводится до 3,000 примеров в Stage-1 и 400 в Stage-2.

---

## Предобработка текста

Используется **лёгкая очистка без лемматизации**:

| Шаг | Описание | Почему |
|---|---|---|
| HTML decode | `&amp;` → `&`, `&lt;` → `<` | соцсети часто содержат HTML |
| Удаление HTML тегов | `<b>текст</b>` → `текст` | — |
| Удаление URL | `https://...` → ` ` | не несут эмоц. нагрузки |
| @mentions | `@user` → ` ` | анонимизация |
| #hashtags | `#радость` → `радость` | сохраняем слово |
| Unicode нормализация | `"текст"` → `"текст"`, `…` → `...` | стандартизация |
| Сжатие повторов | `аааааа` → `ааа`, `!!!` → `!` | шум из соцсетей |
| Whitespace | множественные пробелы → один | — |

**Лемматизация и удаление стоп-слов — НЕ применяются.** BERT-модели обучены на живой морфологии и используют её. Лемматизация снижает F1 на 2–5%.

---

## Аугментация редких классов

Для компенсации дисбаланса (disgust=0.8%, fear=1.5%):

| Метод | Модель | Описание |
|---|---|---|
| **Парафраз** | `cointegrated/rut5-base-paraphraser` | diverse beam search, 3 варианта/предложение |
| **Обратный перевод** | `Helsinki-NLP/opus-mt-ru-en` + `opus-mt-en-ru` | другой словарь через английский pivot |

Аугментация применяется только к train, val/test не затрагиваются.

---

## Архитектура моделей

| Модель | HuggingFace ID | Параметры | Особенности |
|---|---|---|---|
| **ruBERT** | `blanchefort/rubert-base-cased-sentiment` | ~180M | предобучен на RU сентименте |
| **XLM-RoBERTa** | `xlm-roberta-base` | ~278M | 100 языков, лучший кросс-лингвальный перенос |
| **ruBERT-tiny2** | `cointegrated/rubert-tiny2` | ~12M | быстрый, ~85% качества от base |

---

## Стратегия обучения

```
Stage 1 — PRETRAIN
─────────────────────────────────────────────────────────────────
Данные:   stage1_data_augmented (GoEmotions + Dusha + все источники)
Loss:     Focal Loss (γ=2.0) + class weights
LR:       2e-5    Epochs: 3    Batch: 32
Задача:   адаптация к эмоциональной лексике на большом объёме

                         ↓ инициализация весов

Stage 2 — FINE-TUNE
─────────────────────────────────────────────────────────────────
Данные:   stage2_data_augmented (CEDR + BRIGHTER + Aniemore, нативный RU)
Loss:     Cross-Entropy + label smoothing 0.05
LR:       5e-6    Epochs: 3    Batch: 16
Задача:   выравнивание на точные Ekman-метки нативного корпуса
```

---

## Ансамблирование

| Метод | Описание |
|---|---|
| **Soft Voting** | Среднее вероятностей трёх моделей |
| **Weighted Averaging** | Взвешивание по F1-macro |
| **Stacking (LogReg)** | Мета-модель на val_probs (без утечки данных) |
| **Temperature Scaling** | Калибровка уверенности через минимизацию NLL |

---

## Быстрый старт

### Установка

```bash
pip install -r requirements.txt
```

### Запуск пайплайна (Kaggle T4 GPU)

```
01_data_preparation.ipynb   →  сборка и аугментация данных (~20-40 мин)
02_training.ipynb           →  обучение трёх моделей (~2-4 ч на T4)
03_ensemble.ipynb           →  ансамбль и финальная оценка
04_applications.ipynb       →  прикладной анализ текстов
```

### Инференс

```python
from src.inference import EmotionClassifier

clf = EmotionClassifier([
    "results/models/rubert",
    "results/models/xlmroberta",
    "results/models/rubert_tiny",
])

clf.predict("Мне очень страшно идти туда одному")
# [{'fear': 0.71, 'sadness': 0.12, 'anger': 0.06, ...}]

clf.predict_label(["Я так рад! Это лучший день!", "Что за ужас..."])
# ['joy', 'disgust']
```

### Gradio-демо

```bash
python app/app.py --model_dirs results/models/rubert results/models/xlmroberta
```

---

## Структура проекта

```
sentiment-analysis/
│
├── src/
│   ├── data_loader.py      — загрузчики 8+ датасетов, merge_datasets, load_stage2_clean
│   ├── preprocessor.py     — clean_text (HTML, URL, unicode, повторы) — без лемматизации
│   ├── augmentation.py     — TextAugmenter (rut5 паrafраз + back-translation), augment_rare_classes
│   ├── trainer.py          — WeightedTrainer (focal/CE), train_model, train_two_stage
│   ├── ensemble.py         — soft_voting_proba, stacking, fit_temperature, temperature_scaling
│   ├── evaluation.py       — evaluate_predictions, plot_confusion_matrix, compare_models
│   └── inference.py        — EmotionClassifier (batch inference, ансамбль)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   — Блок 1: данные + очистка + аугментация
│   ├── 02_training.ipynb           — Блок 2: двухэтапное обучение ансамбля
│   ├── 03_ensemble.ipynb           — Блок 3: ансамблирование и финальная оценка
│   ├── 04_applications.ipynb       — Блок 4: DH-инструменты и визуализация
│   └── legacy/                     — старые ноутбуки (одиночное обучение, EDA)
│
├── app/
│   └── app.py              — Gradio-демо, совместим с HuggingFace Spaces
│
├── data/                   — локальные датасеты (gitignored)
├── results/                — чекпоинты и результаты экспериментов
└── requirements.txt
```

---

## Прикладные инструменты (DH)

Ноутбук `04_applications.ipynb`:

| Функция | Описание |
|---|---|
| `emotion_arc(text)` | Эмоциональная дуга нарратива (по предложениям через `razdel`) |
| `emotion_timeline(df, date_col)` | Временной ряд эмоций в датированном корпусе |
| `emotion_heatmap(df, group_col)` | Профили по авторам / жанрам / источникам |
| `radar_chart(profiles)` | Паукообразное сравнение нескольких профилей |
| `explain_prediction(text)` | Атрибуция токенов (Integrated Gradients) |
| `emotion_wordclouds(df)` | Облака слов, характерных для каждой эмоции |

---

## Технические детали

### Работа с дисбалансом классов

- **Focal Loss** (γ=2.0): снижает вес лёгких примеров (joy, neutral), усиливает редкие (disgust, fear)
- **Class Weights**: `compute_class_weight("balanced")` — автоматически из распределения
- **MAX_PER_CLASS=15k**: undersampling мажоритарных классов перед аугментацией
- **Аугментация**: доводит disgust/fear/anger до 3k каждый в Stage-1

### Предотвращение утечки данных

- Мета-модель стекинга обучается на `val_probs.npy` (out-of-fold), не на test
- Val/test сплиты **не аугментируются**

---

## Зависимости

```
torch>=2.0, transformers>=4.40, datasets, accelerate   — обучение
scikit-learn, scipy                                     — ансамбль, метрики
pandas, numpy, matplotlib, seaborn                      — анализ
gradio>=4.0                                             — веб-демо
razdel>=0.5                                             — сегментация предложений
sentencepiece, sacremoses                               — аугментация (MarianMT)
peft>=0.6                                               — LoRA (опционально)
transformers-interpret>=0.10                            — объяснимость
pymorphy2, nltk                                         — для классических ML-базелайнов
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
- [ru_go_emotions (seara, 2023)](https://huggingface.co/datasets/seara/ru_go_emotions)
