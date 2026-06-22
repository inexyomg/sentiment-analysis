#!/usr/bin/env python3
"""
Patch 04_applications.ipynb:
  - Replace cell 0 (title + TOC, 7-section table)
  - Replace cell 3 (§1 markdown with key metrics bullets)
  - Insert corpus-stats code cell after cell 4
  - Insert entropy markdown + code cells after the saliency cell (was cell 8, now cell 9)
  - Renumber §4 → §5 in the "Анализ своего текста" markdown
  - Renumber §5 → §6 in the "DH-инструменты" markdown
  - Append §7 DH-applications markdown at the end
"""

import json

NB_PATH = '/home/user/sentiment-analysis/notebooks/04_applications.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }


cells = nb['cells']

# Current structure (13 cells, indices 0-12):
# idx 0  → title markdown
# idx 1  → imports code
# idx 2  → model loading code
# idx 3  → §1 markdown
# idx 4  → load data + distribution plot code
# idx 5  → §2 word clouds markdown
# idx 6  → word clouds code
# idx 7  → §3 markdown
# idx 8  → confidence + saliency code
# idx 9  → §4 markdown (→ §5)
# idx 10 → interactive analysis code
# idx 11 → §5 markdown (→ §6)
# idx 12 → confusion matrix + lex profiles + corpus nav code

# ── 1. Replace cell 0 ─────────────────────────────────────────────────────
cells[0]['source'] = (
    "# Анализ эмоций в русском тексте — DH-инструментарий\n"
    "\n"
    "Этот ноутбук демонстрирует возможности дистиллированной нейросетевой модели для автоматического распознавания эмоций в русскоязычных текстах. Классификация ведётся по **7 категориям Экмана**: гнев · отвращение · страх · радость · грусть · удивление · нейтральность.\n"
    "\n"
    "> **Цель:** сделать работу модели прозрачной и полезной для исследователя-гуманитария.\n"
    "\n"
    "| § | Раздел | Содержание |\n"
    "|---|--------|----------|\n"
    "| **1** | **Тестовый корпус** | Загрузка, статистика, распределение по эмоциям |\n"
    "| **2** | **Словари эмоций** | TF-IDF-облака: характерная лексика каждого класса |\n"
    "| **3** | **Механизм решений** | Уверенность, важность токенов (Gradient × Embedding) |\n"
    "| **4** | **Эмоциональная неоднозначность** | Энтропийный анализ: где граница между эмоциями размыта |\n"
    "| **5** | **Анализ своего текста** | Введите любой текст — модель визуализирует решение |\n"
    "| **6** | **DH-инструментарий** | Матрица ошибок, лексические профили, навигатор корпуса |"
)

# ── 2. Replace cell 3 (§1 markdown) ──────────────────────────────────────
cells[3]['source'] = (
    "## 1. Тестовый корпус\n"
    "\n"
    "Тестовая выборка Stage-2 — нативные русскоязычные тексты (~11k примеров, ~20% выделено в test). Источники: социальные сети, новости, форумы. Предсказания загружаются из `test_preds.npy` дистиллированной модели (если размер совпадает) либо вычисляются на лету через инференс.\n"
    "\n"
    "Ключевые показатели модели на этом корпусе:\n"
    "- **F1-macro** — агрегированное качество по всем 7 классам\n"
    "- **Accuracy** — доля верно определённых эмоций\n"
    "- **Уверенность** — средняя вероятность предсказанного класса"
)

# ── 3. Insert corpus-stats code cell after cell 4 (index 4) ──────────────
CORPUS_STATS_SOURCE = (
    "# ── Статистика корпуса ─────────────────────────────────────────────────────\n"
    "test_df['text_len']   = test_df['text'].str.len()\n"
    "test_df['word_count'] = test_df['text'].str.split().str.len()\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
    "\n"
    "# 1a. Длина текстов\n"
    "axes[0].hist(test_df['text_len'].clip(0, 600), bins=40,\n"
    "             color='#3498db', alpha=0.75, edgecolor='white')\n"
    "axes[0].set_title('Распределение длины текстов')\n"
    "axes[0].set_xlabel('Символов'); axes[0].set_ylabel('Кол-во')\n"
    "axes[0].grid(alpha=0.3)\n"
    "\n"
    "# 1b. Длина по эмоциям (ящик с усами)\n"
    "data_by_emo = [test_df[test_df['true_label'] == e]['word_count'].clip(0, 120).values\n"
    "               for e in label_names]\n"
    "bp = axes[1].boxplot(data_by_emo, labels=label_names, patch_artist=True,\n"
    "                     medianprops=dict(color='white', linewidth=2), flierprops=dict(markersize=3))\n"
    "for patch, emo in zip(bp['boxes'], label_names):\n"
    "    patch.set_facecolor(EMOTION_COLORS[emo]); patch.set_alpha(0.8)\n"
    "axes[1].set_title('Длина текста по эмоциям (слова)')\n"
    "axes[1].set_ylabel('Слов в тексте'); axes[1].tick_params(axis='x', rotation=35)\n"
    "axes[1].grid(axis='y', alpha=0.3)\n"
    "\n"
    "# 1c. Accuracy по классам\n"
    "acc_per_class = test_df.groupby('true_label')['correct'].mean().reindex(label_names)\n"
    "bars = axes[2].bar(label_names, acc_per_class.values,\n"
    "                   color=[EMOTION_COLORS[e] for e in label_names], edgecolor='white')\n"
    "for bar, v in zip(bars, acc_per_class.values):\n"
    "    axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.01,\n"
    "                 f'{v:.0%}', ha='center', fontsize=9)\n"
    "axes[2].axhline(test_df['correct'].mean(), color='black', linestyle='--',\n"
    "                alpha=0.5, label=f'Средняя ({test_df[\"correct\"].mean():.0%})')\n"
    "axes[2].set_ylim(0, 1.15)\n"
    "axes[2].set_title('Точность по классам (Accuracy)')\n"
    "axes[2].set_ylabel('Accuracy'); axes[2].tick_params(axis='x', rotation=35)\n"
    "axes[2].grid(axis='y', alpha=0.3); axes[2].legend(fontsize=9)\n"
    "\n"
    "plt.suptitle('Характеристики тестового корпуса', fontsize=13, y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{WORKING_DIR}/corpus_stats.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "# Сводная таблица\n"
    "_stats = test_df.groupby('true_label').agg(\n"
    "    n=('text', 'count'),\n"
    "    accuracy=('correct', 'mean'),\n"
    "    avg_chars=('text_len', 'mean'),\n"
    "    avg_words=('word_count', 'mean'),\n"
    "    mean_conf=('max_prob', 'mean'),\n"
    ").reindex(label_names).round(3)\n"
    "_stats['accuracy'] = _stats['accuracy'].map('{:.1%}'.format)\n"
    "print('\\nСтатистика по классам:')\n"
    "print(_stats.to_string())"
)

corpus_stats_cell = make_code_cell(CORPUS_STATS_SOURCE)
# Insert after index 4 → new index 5
cells.insert(5, corpus_stats_cell)

# After insertion indices shift by 1:
# idx 0  title (replaced)
# idx 1  imports
# idx 2  model loading
# idx 3  §1 markdown (replaced)
# idx 4  load data + dist plot
# idx 5  NEW corpus stats code
# idx 6  §2 word clouds markdown
# idx 7  word clouds code
# idx 8  §3 markdown
# idx 9  confidence + saliency code  ← insert entropy after this
# idx 10 §4 markdown (→ §5)
# idx 11 interactive code
# idx 12 §5 markdown (→ §6)
# idx 13 confusion matrix

# ── 4. Insert entropy markdown + code after index 9 ──────────────────────
ENTROPY_MD_SOURCE = (
    "## 4. Эмоциональная неоднозначность\n"
    "\n"
    "Некоторые тексты несут несколько эмоций сразу — модель отражает это более равномерным распределением вероятностей. Для измерения многозначности используем **энтропию Шеннона**:\n"
    "\n"
    "$$H = -\\sum_{i} p_i \\log p_i$$\n"
    "\n"
    "- **Низкая энтропия** → одна эмоция явно доминирует, модель уверена\n"
    "- **Высокая энтропия** → вероятности близки → текст эмоционально многозначен\n"
    "\n"
    "Это особенно актуально для литературных и разговорных текстов, где эмоциональный регистр неоднозначен."
)

ENTROPY_CODE_SOURCE = (
    "from scipy.stats import entropy as _scipy_entropy\n"
    "\n"
    "# ── Энтропия предсказаний ──────────────────────────────────────────────────\n"
    "prob_cols = [f'p_{n}' for n in label_names]\n"
    "test_df['entropy'] = test_df[prob_cols].apply(\n"
    "    lambda row: float(_scipy_entropy(row.values + 1e-10)), axis=1\n"
    ")\n"
    "MAX_ENTROPY = float(np.log(len(label_names)))\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
    "\n"
    "# Распределение энтропии по эмоциям\n"
    "for emo in label_names:\n"
    "    vals = test_df[test_df['true_label'] == emo]['entropy']\n"
    "    axes[0].hist(vals, bins=20, alpha=0.5, density=True,\n"
    "                 label=emo, color=EMOTION_COLORS[emo])\n"
    "axes[0].axvline(MAX_ENTROPY, color='black', linestyle='--', alpha=0.7,\n"
    "                label=f'Макс. энтропия ({MAX_ENTROPY:.2f})')\n"
    "axes[0].set_xlabel('Энтропия'); axes[0].set_ylabel('Плотность')\n"
    "axes[0].set_title('Уверенность модели по эмоциям (энтропия)')\n"
    "axes[0].legend(ncol=2, fontsize=8); axes[0].grid(alpha=0.3)\n"
    "\n"
    "# Средняя энтропия по классу\n"
    "mean_ent = test_df.groupby('true_label')['entropy'].mean().reindex(label_names)\n"
    "bars2 = axes[1].bar(label_names, mean_ent.values,\n"
    "                    color=[EMOTION_COLORS[e] for e in label_names], edgecolor='white')\n"
    "for bar, v in zip(bars2, mean_ent.values):\n"
    "    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01,\n"
    "                 f'{v:.2f}', ha='center', fontsize=9)\n"
    "axes[1].set_title('Средняя энтропия по классам')\n"
    "axes[1].set_ylabel('Энтропия Шеннона'); axes[1].tick_params(axis='x', rotation=35)\n"
    "axes[1].grid(axis='y', alpha=0.3)\n"
    "\n"
    "plt.suptitle('Эмоциональная неоднозначность — энтропийный анализ', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig(f'{WORKING_DIR}/emotion_entropy.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "# ── Наиболее неоднозначные тексты ─────────────────────────────────────────\n"
    "print('── Топ-10 эмоционально многозначных текстов (высокая энтропия) ─────')\n"
    "_ambig = test_df.nlargest(10, 'entropy')\n"
    "for _, row in _ambig.iterrows():\n"
    "    _top2 = sorted(\n"
    "        ((n, test_df.loc[row.name, f'p_{n}']) for n in label_names),\n"
    "        key=lambda x: -x[1]\n"
    "    )[:2]\n"
    "    _top2_str = ' + '.join(f'{e}({p:.2f})' for e, p in _top2)\n"
    "    print(f'\\n[{_top2_str} | энтропия={row[\"entropy\"]:.3f} | истинная: {row[\"true_label\"]}]')\n"
    "    print(f'  {row[\"text\"][:180]}')\n"
    "\n"
    "# ── Визуализация токенов для неоднозначных примеров ───────────────────────\n"
    "if clf is not None:\n"
    "    from IPython.display import HTML, display as ipy_display\n"
    "    _ambig_show = test_df.nlargest(4, 'entropy')\n"
    "    ipy_display(HTML('<h4 style=\"margin-top:16px\">Токен-атрибуция для неоднозначных текстов:</h4>'))\n"
    "    for _, row in _ambig_show.iterrows():\n"
    "        ipy_display(HTML(\n"
    "            f'<div style=\"margin:4px 0;font-size:11px;color:#555\">'\n"
    "            f'истинная: <b>{row[\"true_label\"]}</b> · '\n"
    "            f'энтропия: <b>{row[\"entropy\"]:.3f}</b></div>'\n"
    "        ))\n"
    "        show_saliency(row['text'], true_label=row['true_label'])"
)

entropy_md_cell   = make_markdown_cell(ENTROPY_MD_SOURCE)
entropy_code_cell = make_code_cell(ENTROPY_CODE_SOURCE)

# Insert after index 9 → they become 10 and 11
cells.insert(10, entropy_md_cell)
cells.insert(11, entropy_code_cell)

# After two more insertions indices shift by 2:
# idx 0  title (replaced)
# idx 1  imports
# idx 2  model loading
# idx 3  §1 markdown (replaced)
# idx 4  load data + dist plot
# idx 5  corpus stats code  (NEW)
# idx 6  §2 word clouds markdown
# idx 7  word clouds code
# idx 8  §3 markdown
# idx 9  confidence + saliency code
# idx 10 §4 entropy markdown  (NEW)
# idx 11 entropy code         (NEW)
# idx 12 old §4 markdown → §5
# idx 13 interactive code
# idx 14 old §5 markdown → §6
# idx 15 confusion matrix + lex profiles + corpus nav

# ── 5. Renumber §4 → §5 ──────────────────────────────────────────────────
src12 = cells[12]['source']
if isinstance(src12, list):
    src12 = ''.join(src12)
src12 = src12.replace('## 4. Анализ своего текста', '## 5. Анализ своего текста', 1)
cells[12]['source'] = src12

# ── 6. Renumber §5 → §6 ──────────────────────────────────────────────────
src14 = cells[14]['source']
if isinstance(src14, list):
    src14 = ''.join(src14)
src14 = src14.replace('## 5. DH-инструменты', '## 6. DH-инструменты', 1)
cells[14]['source'] = src14

# ── 7. Append §7 DH-applications markdown at the very end ────────────────
DH_APPS_SOURCE = (
    "## 7. DH-приложения: как использовать модель в исследовании\n"
    "\n"
    "### Литературоведение — эмоциональная кривая текста\n"
    "Разбейте роман или поэму на абзацы и прогоните через `clf.predict()` — получите «эмоциональный профиль» нарратива. Сравните эмоциональные дуги разных авторов или эпох.\n"
    "\n"
    "```python\n"
    "paragraphs = [\"Абзац 1...\", \"Абзац 2...\", ...]\n"
    "probs_df   = pd.DataFrame(clf.predict_proba(paragraphs), columns=label_names)\n"
    "probs_df.plot(figsize=(14, 4), alpha=0.7, title=\"Эмоциональная кривая текста\")\n"
    "```\n"
    "\n"
    "### Медиаанализ — мониторинг эмоционального фона\n"
    "Классифицируйте посты/комментарии до и после события, сравните распределения эмоций между группами.\n"
    "\n"
    "### Историческое источниковедение\n"
    "Письма, дневники, протоколы — анализ аффективного регистра документа как атрибутивный признак или инструмент изучения исторической эмоциональности.\n"
    "\n"
    "---\n"
    "\n"
    "### ⚠ Ограничения и рекомендации\n"
    "\n"
    "| Ограничение | Рекомендация |\n"
    "|-------------|-------------|\n"
    "| Обучена на современных текстах | Проверяйте на целевом корпусе перед публикацией |\n"
    "| 7 категорий Экмана — упрощение | Используйте полный вектор вероятностей, не только argmax |\n"
    "| Нет контекста между фрагментами | Анализируйте скользящим окном для длинных текстов |\n"
    "| Ошибки на коротких/разговорных | Проверяйте borderline cases вручную |\n"
    "\n"
    "### Воспроизведение\n"
    "\n"
    "```python\n"
    "from src.inference import EmotionClassifier\n"
    "\n"
    "clf = EmotionClassifier([\"results/models/distilled_xlmr\"])\n"
    "\n"
    "# Один текст\n"
    "result = clf.predict(\"Мне очень страшно идти туда одному\", top_k=None)[0]\n"
    "# → {'fear': 0.71, 'sadness': 0.12, ...}\n"
    "\n"
    "# Батч\n"
    "results = clf.predict([\"текст 1\", \"текст 2\", \"текст 3\"])\n"
    "top_emotions = [max(r, key=r.get) for r in results]\n"
    "```"
)

dh_apps_cell = make_markdown_cell(DH_APPS_SOURCE)
cells.append(dh_apps_cell)

# ── Save ──────────────────────────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

total = len(nb['cells'])
print(f'Saved. Total cells: {total}')
print()
for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    src = c.get('source', '')
    if isinstance(src, list):
        src = ''.join(src)
    first_line = src.split('\n')[0][:90] if src else '(empty)'
    print(f'  [{i:02d}] [{ct[:4]}] {first_line}')
