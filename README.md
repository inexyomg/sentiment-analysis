# Анализ эмоциональной тональности русскоязычных текстов

Полный исследовательский пайплайн для 7-классовой классификации эмоций по таксономии Экмана в русскоязычных текстах. Ансамбль из 7 трансформер-моделей, двухэтапное обучение, прикладные DH-инструменты.

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

## Структура пайплайна

```
┌─────────────────────────────────────────────────────────────────────┐
│  Блок 1: 01_data_preparation.ipynb                                  │
│  8 источников → маппинг в 7 классов → дедупликация → split          │
│  → аугментация редких классов                                        │
│  Выход: stage1_data_augmented / stage2_data_augmented               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Блок 2: 02_training.ipynb                                          │
│  Stage 1: pretrain на большом корпусе (focal loss, lr=2e-5)         │
│  Stage 2: fine-tune на чистом нативном RU (CE+smoothing, lr=5e-6)   │
│  7 моделей: rubert · xlm-roberta · rubert-tiny · rubert-large ·     │
│             ruroberta-large · aniemore-emotion · seara-goem          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Блок 3: 03_ensemble.ipynb                                          │
│  Soft voting · Stacking (LogReg) · Temperature scaling              │
│  Финальная оценка по всем классам                                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Блок 4: 04_applications.ipynb                                      │
│  DH-инструменты: эмоц. дуги, временные ряды, хитмапы               │
│  + Gradio-демо (app/app.py)                                         │
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

Stage-1 использует `method='both'` (паrafраз + обратный перевод) для максимального разнообразия. Focal Loss устойчив к шуму аугментации.

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

## Ансамбль из 7 моделей

| Ключ | HuggingFace ID | Параметры | Особенности |
|---|---|---|---|
| `rubert` | `blanchefort/rubert-base-cased-sentiment` | ~180M | RU сентимент-претрейн |
| `xlmroberta` | `xlm-roberta-base` | ~278M | 100 языков, лучший кросс-лингвальный перенос |
| `rubert_tiny` | `cointegrated/rubert-tiny2` | ~12M | быстрый, ~85% качества base |
| `rubert_large` | `ai-forever/ruBert-large` | ~340M | SberAI, мощная RU база |
| `ruroberta_large` | `ai-forever/ruRoberta-large` | ~355M | SberAI, RoBERTa-архитектура на RU |
| `aniemore_emotion` | `Aniemore/rubert-tiny2-russian-emotion-detection` | ~12M | fine-tuned на CEDR+RESD эмоциях |
| `seara_goem` | `seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions` | ~180M | fine-tuned на GoEmotions RU (28 классов) |

**Batch sizes и gradient accumulation steps** подобраны под VRAM T4 (16GB):
- large-модели: batch=8, grad_accum=4 (эффективный batch=32)
- base-модели: batch=16–32, grad_accum=1–2
- tiny-модели: batch=64, grad_accum=1

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

---

## Ансамблирование (Блок 3)

| Метод | Описание |
|---|---|
| **Soft Voting** | Среднее вероятностей всех 7 моделей |
| **Weighted Averaging** | Взвешивание по F1-macro каждой модели |
| **Stacking (LogReg)** | Мета-модель на val_probs (out-of-fold, без утечки) |
| **Temperature Scaling** | Калибровка уверенности: минимизация NLL на val |

---

## Прикладные инструменты (Блок 4)

| Функция | Описание |
|---|---|
| `emotion_arc(text)` | Эмоциональная дуга нарратива по предложениям (`razdel`) |
| `emotion_timeline(df, date_col)` | Временной ряд эмоций в датированном корпусе |
| `emotion_heatmap(df, group_col)` | Профили по авторам / жанрам / источникам |
| `radar_chart(profiles)` | Паукообразное сравнение нескольких профилей |
| `explain_prediction(text)` | Атрибуция токенов (Integrated Gradients) |
| `emotion_wordclouds(df)` | Облака слов, характерных для каждой эмоции |

---

## Запуск на Kaggle

Платформа: **Kaggle Notebook, T4 x2 GPU (16 GB VRAM каждый).**

Репозиторий подключён как Kaggle Dataset: `/kaggle/input/datasets/inexyy/se-analysis`.

### Порядок запуска

```
1. 01_data_preparation.ipynb  — сборка, очистка, дедупликация, аугментация (~20-40 мин)
2. 02_training.ipynb          — обучение 7 моделей (~3-6 ч на T4 x2)
3. 03_ensemble.ipynb          — ансамбль и финальная оценка
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

# Инференс
python -c "
from src.inference import EmotionClassifier
clf = EmotionClassifier(['results/models/rubert', 'results/models/xlmroberta'])
print(clf.predict('Мне очень страшно идти туда одному'))
"

# Gradio-демо
python app/app.py --model_dirs results/models/rubert results/models/xlmroberta
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
│   ├── ensemble.py         — soft_voting, weighted_avg, stacking, temperature_scaling
│   ├── evaluation.py       — evaluate_predictions, confusion_matrix, compare_models
│   └── inference.py        — EmotionClassifier (batch + ансамбль)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   — Блок 1: данные, дедуп, аугментация
│   ├── 02_training.ipynb           — Блок 2: двухэтапное обучение 7 моделей
│   ├── 03_ensemble.ipynb           — Блок 3: ансамблирование, финальная оценка
│   └── 04_applications.ipynb       — Блок 4: DH-инструменты и визуализация
│
├── data/
│   ├── stage1_data/                — HuggingFace DatasetDict (Arrow)
│   ├── stage1_data_augmented/      — Stage-1 + аугментация
│   ├── stage2_data/                — Stage-2 нативный RU
│   └── stage2_data_augmented/      — Stage-2 + аугментация
│
├── app/
│   └── app.py              — Gradio-демо (HuggingFace Spaces совместим)
│
├── results/                — чекпоинты моделей и результаты (gitignored)
└── requirements.txt
```

---

## Ключевые технические решения

| Решение | Причина |
|---|---|
| GoEmotions `simplified` вместо `raw` | `raw` даёт 3.6× дублей текста с меткой одного аннотатора после дедупа |
| Dusha исключена | Аннотация по голосу, не по тексту — «Какое время» → sadness |
| Дедупликация до сплита | Без неё 43% test оказывались в train (утечка данных) |
| `repetition_penalty=1.2` (не 3.0) | 3.0 ломало rut5 и давало нечитаемый мусор |
| `_is_valid_ru()` фильтр | Отсеивает аугментацию с недостаточным % кириллицы |
| Stage-2 только `backtranslation` | Паrafраз может сменить эмоциональный тон; обратный перевод стабильнее |
| MAX_PER_CLASS=35,000 | Caps мажоритарный joy (51k) до разумного уровня, не уничтожая данные |
| Focal Loss Stage-1 | Устойчив к шуму аугментации, фокусируется на редких классах |
| CE + smoothing Stage-2 | Меньший LR + smoothing 0.05 — мягкая калибровка без переобучения |

---

## Зависимости

```
torch>=2.0, transformers>=4.40, datasets, accelerate   — обучение
scikit-learn, scipy                                     — ансамбль, метрики
pandas, numpy, matplotlib, seaborn                      — анализ
gradio>=4.0                                             — веб-демо
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
