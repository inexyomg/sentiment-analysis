# Анализ эмоциональной тональности русскоязычных текстов цифровыми методами

*Выпускная квалификационная работа (магистратура) · Цифровые методы в гуманитарных науках.*

Полный воспроизводимый исследовательский пайплайн для 7-классовой классификации эмоций по таксономии Экмана в русскоязычных текстах: **ансамбль из 3 трансформеров-учителей**, двухэтапное обучение, **дистилляция знаний в одну компактную модель-студент** и прикладные DH-инструменты.

---

## Таксономия эмоций (Ekman, 7 классов, single-label)

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

## Структура пайплайна

```
┌─────────────────────────────────────────────────────────────────────┐
│  Блок 1: 01_data_preparation.ipynb                                  │
│  8 источников → маппинг в 7 классов → дедупликация → split          │
│  → аугментация редких классов                                        │
│  Выход: stage1_data_augmented / stage2_data_augmented               │
│  Графики: distribution_before_aug.png · s1_augmentation.png         │
│           s2_augmentation.png                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Блок 2: 02_training.ipynb                                          │
│  Stage 1: pretrain на большом корпусе (focal loss, lr=2e-5)         │
│  Stage 2: fine-tune на чистом нативном RU (CE+smoothing, lr=5e-6)   │
│  3 модели-учителя: ruroberta_large · xlmroberta · seara_goem         │
│  Выход (на модель): test/val probs·preds·labels.npy + results.json  │
│  Графики: two_stage_comparison.png · per_class_f1_two_stage.png      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Блок 3: 03_ensemble.ipynb                                          │
│  §1–5: Hard/Soft/Weighted Voting · Stacking · Temperature Scaling   │
│        Финальная оценка · Сохранение лучшей модели-ансамбля         │
│  §6:   Knowledge Distillation — 3 учителя → 1 студент ruBert-large  │
│        Loss = α·KL/T² + (1−α)·CE  (двухэтапная: T=4.0 → T=2.0)      │
│  Графики: model_comparison.png · cm_best_ensemble.png               │
│           distillation_training.png                                  │
│  JSON/pkl: final_summary.json · ensemble/ · distillation_results    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Блок 4: 04_applications.ipynb                                      │
│  DH-инструменты: облака слов, энтропийный анализ, важность токенов, │
│  матрица ошибок, лексические профили, навигатор корпуса             │
│  Графики: emotion_wordclouds.png · emotion_entropy.png · …          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Датасеты

### Stage 1 — большой смешанный корпус

| Датасет | Примеров | Тип | Примечание |
|---|---|---|---|
| `seara/ru_go_emotions` **simplified** | ~54k | перевод EN→RU | Консенсус-метки аннотаторов, уникальные тексты. `simplified` — агрегированные метки; `raw` даёт 3.6× дублей текста с конфликтующими метками |
| `Djacon/ru-izard-emotions` | ~30k | перевод RU Reddit | 7 из 10 эмоций Изарда → Экман |
| `Aniemore/cedr-m7` | ~11k | **нативный RU** | 7 классов Экман включая disgust и neutral |
| `brighter-dataset/BRIGHTER-emotion-categories` (rus) | ~5k | **нативный RU** | SemEval-2025 Task 11, Toloka-аннотация |
| `Aniemore/resd_annotated` | ~1.7k | **нативный RU** | STT-транскрипты, 7 классов |
| `Helsinki-NLP/XED` (ru) | ~2.4k | субтитры RU | 8 эмоций Plutchik → Ekman |
| `sismetanin/rureviews` | ~90k | отзывы Wildberries | sentiment pos/neg → joy/sadness/neutral |
| `sismetanin/rusentitweet` | ~13k | твиты | sentiment pos/neg/neutral → joy/sadness/neutral |

> **Почему нет Dusha:** Dusha аннотирует эмоцию по голосу говорящего, а не по смыслу текста. «Какое сейчас время», сказанное грустным голосом → sadness. Для текстового классификатора это шум — Dusha исключена.

> **Почему GoEmotions simplified:** `raw` содержит по одной строке на каждого аннотатора (~3.6 копии текста). После дедупликации случайно сохраняется метка одного аннотатора. `simplified` — уже агрегированный консенсус, тексты уникальны.

**Распределение Stage-1 до аугментации (total 149k, после cap MAX_PER_CLASS=35k):**

| Класс | Всего | После cap |
|---|---|---|
| joy | 51,660 | 35,000 (обрезан) |
| neutral | 34,912 | 34,912 |
| surprise | 22,148 | 22,148 |
| sadness | 15,618 | 15,618 |
| anger | 12,722 | 12,722 |
| fear | 6,428 | 6,428 |
| disgust | 5,531 | 5,531 |

### Stage 2 — чистый нативный корпус

Только нативные RU датасеты с качественной ручной разметкой:

| Датасет | Train | Назначение |
|---|---|---|
| `Aniemore/cedr-m7` | ~9k | основной нативный источник, 7 классов |
| `brighter-dataset/BRIGHTER-emotion-categories` | ~3.5k | нативный, Toloka, 6 классов |
| `Aniemore/resd_annotated` | ~1.2k | нативный, STT |

**Распределение Stage-2 train до аугментации:** anger 1,301 · disgust 588 · fear 1,246 · joy 3,355 · sadness 2,091 · surprise 300 · neutral 4,155

---

## Предобработка текста

Лёгкая очистка **без лемматизации** — BERT-модели обучены на живой морфологии, лемматизация снижает F1 на 2–5%:

| Шаг | Что делает |
|---|---|
| HTML decode | `&amp;` → `&`, `&lt;` → `<` |
| Удаление HTML тегов | `<b>текст</b>` → `текст` |
| Удаление URL | `https://...` → пробел |
| @mentions | `@user` → пробел |
| #hashtags | `#радость` → `радость` (слово сохраняется) |
| Unicode нормализация | типографские кавычки, `…` → `...` |
| Сжатие повторов | `аааааа` → `ааа`, `!!!` → `!` |
| Whitespace | множественные пробелы → один |

---

## Дедупликация и предотвращение утечки

`merge_datasets()` в `src/data_loader.py` выполняет глобальную дедупликацию **перед** сплитом:

```python
full_df["_key"] = full_df["text"].str.strip().str.lower()
full_df = full_df.drop_duplicates(subset=["_key"])
# затем train_test_split
```

**Проблема без дедупликации:** разные источники содержат одинаковые тексты в разных сплитах. При конкатенации без дедупа 43% тестовых примеров Stage-1 оказывались в train → мнимо высокие метрики.

Также: val/test **никогда не аугментируются** — только train.

---

## Аугментация редких классов

### Stage 1 (`AUG_METHOD_S1 = 'both'`, цель 15,000/класс в train)

| Класс | Train до | Train после |
|---|---|---|
| anger | ~8,905 | 15,000 |
| fear | ~4,500 | 15,000 |
| disgust | ~3,872 | 15,000 |
| sadness | ~10,933 | 15,000 |

Stage-1 использует `method='both'` (парафраз + обратный перевод) для максимального разнообразия. Focal Loss устойчив к шуму аугментации.

### Stage 2 (`AUG_METHOD_S2 = 'backtranslation'`, цель 1,200/класс в train)

| Класс | Train до | Train после |
|---|---|---|
| disgust | 588 | 1,200 |
| surprise | 300 | 1,200 |

Stage-2 использует только `backtranslation` — обратный перевод точнее сохраняет эмоциональный тон, что критично для fine-tuning стейджа.

### Методы аугментации

| Метод | Модель | Описание |
|---|---|---|
| **Парафраз** | `cointegrated/rut5-base-paraphraser` | sampling (t=0.7, top_p=0.9), repetition_penalty=1.2 |
| **Обратный перевод** | `Helsinki-NLP/opus-mt-ru-en` + `opus-mt-en-ru` | RU→EN→RU, лексическое разнообразие через pivot |

После генерации каждый пример проходит фильтр `_is_valid_ru()`: минимум 60% кириллических символов, ≥2 реальных слова длиной ≥3 буквы. Это отсеивает мусорные выходы генерации.

---

## Ансамбль из 3 моделей-учителей + дистилляция в студента

Гетерогенный ансамбль из трёх взаимодополняющих трансформеров (разные домены и языки предобучения), знания которого затем сжимаются в одну модель-студент через дистилляцию.

### Модели-учителя

| Ключ | HuggingFace ID | Параметры | Особенности |
|---|---|---|---|
| `ruroberta_large` | `ai-forever/ruRoberta-large` | ~355M | SberAI, RoBERTa-архитектура, сильный RU-претрейн |
| `xlmroberta` | `xlm-roberta-base` | ~278M | 100 языков, кросс-лингвальный перенос |
| `seara_goem` | `seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions` | ~180M | fine-tuned на GoEmotions RU |

### Модель-студент

| Ключ | HuggingFace ID | Параметры | Особенности |
|---|---|---|---|
| `distilled_rubert_large` | `ai-forever/ruBert-large` | ~427M | та же архитектура, что у учителя ruRoberta-large; обучается на мягких метках ансамбля |

Дистилляция позволяет получить **одну** модель уровня всего ансамбля (см. раздел «Результаты»): не нужно держать в памяти и инференсить три модели сразу.

---

## Стратегия двухэтапного обучения

```
Stage 1 — PRETRAIN на большом смешанном корпусе
──────────────────────────────────────────────────────
Данные:   stage1_data_augmented (~124k train после aug)
Loss:     Focal Loss (γ=2.0) + class weights
LR:       2e-5    Epochs: 3    FP16: True
MAX_LEN:  128 токенов
Задача:   широкий эмоциональный словарь, устойчивость к шуму

                     ↓ веса Stage 1 → инициализация Stage 2

Stage 2 — FINE-TUNE на чистом нативном RU
──────────────────────────────────────────────────────
Данные:   stage2_data_augmented (~14.5k train после aug)
Loss:     CrossEntropy + label smoothing 0.05
LR:       5e-6    Epochs: 3    FP16: True
MAX_LEN:  128 токенов
Задача:   убрать «акцент» переводных данных, точная Ekman-разметка
```

Двухэтапный подход даёт +2–4% F1-macro по сравнению с обучением только на объединённых данных.

**Batch sizes и gradient accumulation steps** подобраны под VRAM T4 (16 GB):
- large-модели (ruroberta_large, ruBert-large): batch=8–16, grad_accum=2–4, gradient checkpointing
- base-модели (xlmroberta, seara_goem): batch=16–32, grad_accum=1–2

---

## Ансамблирование (Блок 3)

| Метод | Описание |
|---|---|
| **Hard Voting** | Голосование по предсказанным меткам (majority vote) |
| **Soft Voting** | Среднее вероятностей всех 3 моделей |
| **Weighted Averaging** | Взвешивание по F1-macro каждой модели |
| **Stacking (LogReg)** | Линейная мета-модель на val_probs (out-of-fold, без утечки) |
| **Stacking (SVM / XGBoost / GradientBoosting)** | Нелинейные мета-модели; улавливают взаимодействия между 21 входной вероятностью (3 модели × 7 классов) |
| **Temperature Scaling** | Калибровка уверенности: минимизация NLL на val |

Все stacking-варианты используют единый API:
```python
stacking_ensemble(val_probs, val_labels, test_probs, meta_learner='logistic')
# meta_learner: 'logistic' | 'svm' | 'xgboost' | 'gradient_boosting'
```

---

## Дистилляция знаний (Блок 3, §6)

Три учителя усредняют свои вероятности → получаются **мягкие метки**, на которых обучается студент `ruBert-large`. Дистилляция двухэтапная — по аналогии с обучением учителей:

```
3 учителя → avg(probs) = мягкие метки
                  │
                  ▼
       Студент: ruBert-large

STAGE-1D · 84k примеров        STAGE-2D · 11k нативных RU
широкое знание                 точная Ekman-разметка
T=4.0 · α=0.9 · lr=5e-5        T=2.0 · α=0.7 · lr=2e-5

LOSS = α · T² · KL(student/T ‖ teacher/T) + (1−α) · CE
```

Один студент достигает качества всего ансамбля **при одной модели вместо трёх**.

---

## Результаты

Метрики на едином тестовом наборе (нативный RU, Stage-2 test).

### Одиночные модели

| Модель | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| ruRoBERTa-large | 0.860 | **0.829** | 0.860 |
| seara-goem | 0.816 | 0.762 | 0.817 |
| XLM-RoBERTa | 0.804 | 0.742 | 0.805 |

### Ансамбль (voting)

| Метод | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| Weighted Averaging | 0.848 | 0.817 | 0.848 |
| Soft Voting | 0.847 | 0.815 | 0.847 |
| Hard Voting | 0.838 | 0.795 | 0.840 |

### Стекинг

| Метод | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| Stacking LogReg | 0.854 | **0.820** | 0.854 |
| Stacking SVM | 0.850 | 0.819 | 0.849 |
| Stacking GradBoost | 0.848 | 0.809 | 0.849 |
| Stacking XGBoost | 0.850 | 0.806 | 0.850 |

### Дистилляция

| Модель | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| **Distilled (ruBert-large)** | **0.863** | **0.842** | **0.868** |

**Итог:** дистиллированный студент `ruBert-large` (F1-macro **0.842**) превосходит и лучший ансамбль-стекинг (0.820), и сильнейшую одиночную модель ruRoBERTa-large (0.829) — при одной модели в инференсе вместо трёх.

---

## Запуск на Kaggle

Платформа: **Kaggle Notebook, T4 x2 GPU (16 GB VRAM каждый).**

Репозиторий подключён как Kaggle Dataset: `/kaggle/input/datasets/inexyy/se-analysis`.

### Порядок запуска

```
1. 01_data_preparation.ipynb  — сборка, очистка, дедупликация, аугментация (~20-40 мин)
2. 02_training.ipynb          — обучение 3 моделей-учителей (~3-6 ч на T4 x2)
3. 03_ensemble.ipynb          — ансамбль, дистилляция, финальная оценка
4. 04_applications.ipynb      — DH-инструменты
```

### Важно: read-only input

`/kaggle/input/` — read-only. Ноутбуки автоматически копируют данные в `/kaggle/working/` перед использованием `load_from_disk()`.

Готовые датасеты уже лежат в `data/` репозитория и доступны без пересборки:

```
data/
├── stage1_data/              # Stage-1 до аугментации
├── stage1_data_augmented/    # Stage-1 после аугментации
├── stage2_data/              # Stage-2 до аугментации
└── stage2_data_augmented/    # Stage-2 после аугментации
```

---

## Локальный запуск

```bash
pip install -r requirements.txt

# Инференс через ансамбль учителей (soft voting)
python -c "
from src.inference import EmotionClassifier
clf = EmotionClassifier(['results/models/ruroberta_large', 'results/models/xlmroberta', 'results/models/seara_goem'])
print(clf.predict('Мне очень страшно идти туда одному'))
"

# Инференс через сохранённый финальный ансамбль (после 03_ensemble.ipynb)
python -c "
from src.inference import EmotionClassifier
clf = EmotionClassifier.from_config('results/ensemble')
print(clf.predict_label(['Мне очень страшно идти туда одному']))
# ['fear']
"
```

---

## Структура проекта

```
sentiment-analysis/
│
├── src/
│   ├── data_loader.py      — загрузчики 8 датасетов, merge_datasets (с дедуп),
│   │                         load_stage2_clean, маппинг → 7 классов Экмана
│   ├── preprocessor.py     — clean_text: HTML, URL, unicode, повторы (без лемм)
│   ├── augmentation.py     — TextAugmenter (rut5 + MarianMT), _is_valid_ru фильтр,
│   │                         augment_rare_classes (раздельные методы S1/S2)
│   ├── trainer.py          — WeightedTrainer (focal/CE+smoothing), train_two_stage
│   ├── ensemble.py         — voting, stacking, temperature scaling, дистилляция,
│   │                         save_ensemble / load_ensemble_config
│   ├── evaluation.py       — evaluate_predictions, confusion_matrix, compare_models
│   └── inference.py        — EmotionClassifier (batch + ансамбль + from_config)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   — Блок 1: данные, дедуп, аугментация
│   ├── 02_training.ipynb           — Блок 2: двухэтапное обучение 3 учителей
│   ├── 03_ensemble.ipynb           — Блок 3: ансамблирование, дистилляция, оценка
│   └── 04_applications.ipynb       — Блок 4: DH-инструменты и визуализация
│
├── data/
│   ├── stage1_data/                — HuggingFace DatasetDict (Arrow)
│   ├── stage1_data_augmented/      — Stage-1 + аугментация
│   ├── stage2_data/                — Stage-2 нативный RU
│   └── stage2_data_augmented/      — Stage-2 + аугментация
│
├── results/                — чекпоинты моделей и результаты (gitignored)
│   ├── models/
│   │   └── {model_key}/    — веса + токенизатор (Stage-2 финал)
│   ├── ensemble/           — финальный ансамбль
│   └── …                   — отчёты и графики (см. Выходные файлы)
└── requirements.txt
```

---

## Выходные файлы

Все файлы пишутся в `WORKING_DIR` (`/kaggle/working` на Kaggle, `results/` локально).

### Блок 1 — подготовка данных

| Файл | Что содержит |
|---|---|
| `distribution_before_aug.png` | Гистограмма классов Stage-1 train до аугментации |
| `s1_augmentation.png` | Сравнение распределения до/после аугментации Stage-1 |
| `s2_augmentation.png` | Сравнение распределения до/после аугментации Stage-2 |

### Блок 2 — обучение (создаётся для каждой из 3 моделей-учителей)

Путь: `models/{model_key}/` (`models/ruroberta_large/`, `models/xlmroberta/`, `models/seara_goem/`)

| Файл | Что содержит |
|---|---|
| `test_probs.npy` | Матрица вероятностей на тест. выборке — shape `(N, 7)` |
| `test_preds.npy` | Argmax-предсказания на тест. выборке — shape `(N,)` |
| `test_labels.npy` | Истинные метки тест. выборки — shape `(N,)` |
| `val_probs.npy` | Матрица вероятностей на val-выборке — shape `(M, 7)` |
| `val_preds.npy` | Argmax-предсказания на val-выборке — shape `(M,)` |
| `val_labels.npy` | Истинные метки val-выборки — shape `(M,)` |
| `results.json` | Accuracy, F1-macro/weighted, полный `classification_report` по классам |
| `config.json` + веса | Стандартный HuggingFace checkpoint для загрузки через `from_pretrained` |

Общие файлы Блока 2:

| Файл | Что содержит |
|---|---|
| `label_names.json` | Список `["anger","disgust","fear","joy","sadness","surprise","neutral"]` |
| `ensemble_config.json` | Пути к Stage-2 директориям + гиперпараметры обучения |
| `two_stage_comparison.png` | Столбиковый график F1-macro Stage 1 vs Stage 2 по всем моделям |
| `per_class_f1_two_stage.png` | Per-class F1 после двухэтапного обучения |

### Блок 3 — ансамблирование и дистилляция

| Файл | Что содержит |
|---|---|
| `model_comparison.png` | F1-macro лучшего ансамбля и всех индивидуальных моделей (bar chart) |
| `cm_best_ensemble.png` | Матрица ошибок лучшего ансамбля — 7×7, нормализованная |
| `final_summary.json` | Все метрики: индивидуальные модели, все voting-методы, все stacking-методы; лучший метод и его F1 |
| `ensemble/ensemble_config.json` | Конфиг финального ансамбля: метод, пути к моделям, веса, label_names, F1 |
| `ensemble/meta_learner.pkl` | Обученная мета-модель sklearn (только для stacking-вариантов) |
| `distill_soft_labels_train.npy` | Кеш мягких меток учителей на Stage-2 train — shape `(N_train, 7)` |
| `distillation_training.png` | Кривые потерь (total/KL/CE) и F1-macro по эпохам |
| `distillation_results.json` | Гиперпараметры, best_epoch, финальные метрики, история обучения |
| `models/distilled_rubert_large/` | Готовый HuggingFace checkpoint студента |

Структура `final_summary.json`:
```json
{
  "task": "emotion classification (Ekman 7-class)",
  "label_names": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
  "individual_models": { "accuracy": {…}, "f1_macro": {…}, "f1_weighted": {…} },
  "ensemble_methods":  { "accuracy": {…}, "f1_macro": {…}, "f1_weighted": {…} },
  "stacking_methods":  { "accuracy": {…}, "f1_macro": {…}, "f1_weighted": {…} },
  "best_ensemble": "Stacking LogReg",
  "best_f1_macro": 0.820
}
```

Загрузка дистиллированной модели:
```python
from src.inference import EmotionClassifier
clf = EmotionClassifier("results/models/distilled_rubert_large")
clf.predict_label(["Мне очень страшно идти туда одному"])
# ['fear']
```

### Блок 4 — DH-инструменты

| Файл | Что содержит |
|---|---|
| `test_distribution.png` | Распределение истинных и предсказанных меток на тесте |
| `corpus_stats.png` | Длина текстов, длина по эмоциям, точность по классам |
| `emotion_wordclouds.png` | TF-IDF-облака слов, характерных для каждой из 7 эмоций |
| `confidence_dist.png` | Распределение уверенности модели по классам |
| `emotion_entropy.png` | Энтропийный анализ эмоциональной неоднозначности |
| `dh_confusion.png` | Матрица ошибок на тестовом наборе |
| `dh_lexprofile.png` | Лексические профили эмоций (средний TF-IDF) |

---

## Ключевые технические решения

| Решение | Причина |
|---|---|
| GoEmotions `simplified` вместо `raw` | `raw` даёт 3.6× дублей текста с меткой одного аннотатора после дедупа |
| Dusha исключена | Аннотация по голосу, не по тексту — «Какое время» → sadness |
| Дедупликация до сплита | Без неё 43% test оказывались в train (утечка данных) |
| `repetition_penalty=1.2` (не 3.0) | 3.0 ломало rut5 и давало нечитаемый мусор |
| `_is_valid_ru()` фильтр | Отсеивает аугментацию с недостаточным % кириллицы |
| Stage-2 только `backtranslation` | Парафраз может сменить эмоциональный тон; обратный перевод стабильнее |
| MAX_PER_CLASS=35,000 | Caps мажоритарный joy (51k) до разумного уровня, не уничтожая данные |
| Focal Loss Stage-1 | Устойчив к шуму аугментации, фокусируется на редких классах |
| CE + smoothing Stage-2 | Меньший LR + smoothing 0.05 — мягкая калибровка без переобучения |
| Дистилляция в `ruBert-large` | Качество ансамбля в одной модели; gradient checkpointing под VRAM T4 |

---

## Зависимости

```
torch>=2.0, transformers>=4.40, datasets, accelerate   — обучение
scikit-learn, scipy                                     — ансамбль, метрики
xgboost>=1.7                                            — XGBoost мета-ученик
pandas, numpy, matplotlib, seaborn                      — анализ
wordcloud, pymorphy2                                    — облака слов, лемматизация (DH)
razdel>=0.5                                             — сегментация предложений
sentencepiece, sacremoses                               — MarianMT токенизация
py7zr                                                   — работа с архивами
```

---

## Ссылки

- [GoEmotions (Google, 2020)](https://arxiv.org/abs/2005.00547)
- [CEDR-M7 (Aniemore)](https://huggingface.co/datasets/Aniemore/cedr-m7)
- [ru_go_emotions (seara, 2023)](https://huggingface.co/datasets/seara/ru_go_emotions)
- [BRIGHTER (SemEval-2025 Task 11)](https://arxiv.org/abs/2502.11926)
- [XED — Cross-lingual Emotion Dataset (Helsinki-NLP)](https://github.com/Helsinki-NLP/XED)
- [Aniemore — Russian Emotional AI](https://huggingface.co/Aniemore)
- [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [Temperature Scaling (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
