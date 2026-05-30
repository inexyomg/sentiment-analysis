"""
Restart Stage 2 training for a single model after a disk-full crash.

Usage (in Kaggle notebook cell):
    %run /kaggle/input/datasets/inexyy/se-analysis/scripts/resume_stage2.py

Or with a different model key:
    MODEL_KEY = 'rubert_large'
    %run /kaggle/input/datasets/inexyy/se-analysis/scripts/resume_stage2.py

Requires that Stage 1 for the target model is already complete
(stage1_dir must contain config.json and model weights).
"""

import sys, os, shutil, glob

# ── Environment ────────────────────────────────────────────────────────────
PROJECT_ROOT = '/kaggle/input/datasets/inexyy/se-analysis' if os.path.exists('/kaggle') else os.path.abspath('..')
WORKING_DIR  = '/kaggle/working' if os.path.exists('/kaggle') else '../results'

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Config (mirrors 02_training.ipynb) ────────────────────────────────────
# Override MODEL_KEY before %run to target a different model
if 'MODEL_KEY' not in dir():
    MODEL_KEY = 'ruroberta_large'

MODELS = {
    'rubert': {
        'model_name': 'blanchefort/rubert-base-cased-sentiment',
        'stage1_dir': f'{WORKING_DIR}/models/rubert_s1',
        'stage2_dir': f'{WORKING_DIR}/models/rubert',
        's2_batch_size': 32, 's2_gradient_accumulation_steps': 1,
    },
    'xlmroberta': {
        'model_name': 'xlm-roberta-base',
        'stage1_dir': f'{WORKING_DIR}/models/xlmroberta_s1',
        'stage2_dir': f'{WORKING_DIR}/models/xlmroberta',
        's2_batch_size': 16, 's2_gradient_accumulation_steps': 2,
    },
    'rubert_tiny': {
        'model_name': 'cointegrated/rubert-tiny2',
        'stage1_dir': f'{WORKING_DIR}/models/rubert_tiny_s1',
        'stage2_dir': f'{WORKING_DIR}/models/rubert_tiny',
        's2_batch_size': 64, 's2_gradient_accumulation_steps': 1,
    },
    'rubert_large': {
        'model_name': 'ai-forever/ruBert-large',
        'stage1_dir': f'{WORKING_DIR}/models/rubert_large_s1',
        'stage2_dir': f'{WORKING_DIR}/models/rubert_large',
        's2_batch_size': 8, 's2_gradient_accumulation_steps': 4,
    },
    'ruroberta_large': {
        'model_name': 'ai-forever/ruRoberta-large',
        'stage1_dir': f'{WORKING_DIR}/models/ruroberta_large_s1',
        'stage2_dir': f'{WORKING_DIR}/models/ruroberta_large',
        's2_batch_size': 8, 's2_gradient_accumulation_steps': 4,
    },
    'aniemore_emotion': {
        'model_name': 'Aniemore/rubert-tiny2-russian-emotion-detection',
        'stage1_dir': f'{WORKING_DIR}/models/aniemore_emotion_s1',
        'stage2_dir': f'{WORKING_DIR}/models/aniemore_emotion',
        's2_batch_size': 64, 's2_gradient_accumulation_steps': 1,
    },
    'seara_goem': {
        'model_name': 'seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions',
        'stage1_dir': f'{WORKING_DIR}/models/seara_goem_s1',
        'stage2_dir': f'{WORKING_DIR}/models/seara_goem',
        's2_batch_size': 32, 's2_gradient_accumulation_steps': 1,
    },
}

S2_EPOCHS    = 3
S2_LR        = 5e-6
S2_LOSS      = 'ce'
S2_SMOOTHING = 0.05
MAX_LENGTH   = 128
FP16         = True
SEED         = 42

# ── Validate ───────────────────────────────────────────────────────────────
cfg = MODELS[MODEL_KEY]
s1_dir = cfg['stage1_dir']
s2_dir = cfg['stage2_dir']

if not os.path.isfile(os.path.join(s1_dir, 'config.json')):
    raise FileNotFoundError(
        f"Stage-1 checkpoint not found: {s1_dir}\n"
        "Make sure Stage 1 completed successfully before running this script."
    )

print(f"Resuming Stage 2 for: {MODEL_KEY}")
print(f"  Stage-1 dir : {s1_dir}")
print(f"  Stage-2 dir : {s2_dir}")

# ── Free disk space ────────────────────────────────────────────────────────
print("\nFreeing disk space...")

# Remove checkpoint-* dirs from ALL models (intermediate state, no longer needed)
freed = 0
for key, c in MODELS.items():
    for d in [c['stage1_dir'], c['stage2_dir']]:
        for ckpt in glob.glob(os.path.join(d, 'checkpoint-*')):
            size = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(ckpt)
                for f in files
            ) / 1024**3
            shutil.rmtree(ckpt, ignore_errors=True)
            print(f"  Removed {ckpt}  ({size:.2f} GB)")
            freed += size

print(f"Total freed: {freed:.2f} GB")

# Remove failed/partial Stage-2 dir for target model (clean slate)
if os.path.exists(s2_dir):
    shutil.rmtree(s2_dir)
    print(f"  Removed partial Stage-2: {s2_dir}")
os.makedirs(s2_dir, exist_ok=True)

# ── Load Stage-2 dataset ───────────────────────────────────────────────────
from datasets import load_from_disk

s2_aug_path = f'{WORKING_DIR}/stage2_data_augmented'
s2_raw_path = f'{WORKING_DIR}/stage2_data'

if os.path.isdir(s2_aug_path):
    stage2_ds = load_from_disk(s2_aug_path)
    print(f"\nLoaded Stage-2 augmented: {s2_aug_path}")
elif os.path.isdir(s2_raw_path):
    stage2_ds = load_from_disk(s2_raw_path)
    print(f"\nLoaded Stage-2 raw: {s2_raw_path}")
else:
    # Try copying from git input
    import shutil as _sh
    _data_src = os.path.join(PROJECT_ROOT, 'data', 'stage2_data_augmented')
    if os.path.isdir(_data_src):
        _sh.copytree(_data_src, s2_aug_path)
        stage2_ds = load_from_disk(s2_aug_path)
        print(f"\nCopied + loaded Stage-2 from git input: {_data_src}")
    else:
        raise FileNotFoundError(
            "Stage-2 dataset not found in WORKING_DIR or git input.\n"
            "Run 01_data_preparation.ipynb first, or copy data manually."
        )

print("Stage-2 splits:")
for split in stage2_ds:
    print(f"  {split:12s}: {len(stage2_ds[split]):,}")

# ── Run Stage 2 ────────────────────────────────────────────────────────────
from src.data_loader import EKMAN_LABEL_NAMES
from src.trainer import train_model

LABEL_NAMES = EKMAN_LABEL_NAMES
NUM_LABELS  = len(LABEL_NAMES)

print(f"\n{'='*60}")
print(f"STAGE 2 — {MODEL_KEY}")
print(f"{'='*60}")

_, _, results2 = train_model(
    model_name=s1_dir,
    dataset=stage2_ds,
    output_dir=s2_dir,
    num_labels=NUM_LABELS,
    label_names=LABEL_NAMES,
    num_epochs=S2_EPOCHS,
    batch_size=cfg['s2_batch_size'],
    learning_rate=S2_LR,
    max_length=MAX_LENGTH,
    fp16=FP16,
    seed=SEED,
    loss_type=S2_LOSS,
    label_smoothing=S2_SMOOTHING,
    use_class_weights=False,
    gradient_accumulation_steps=cfg['s2_gradient_accumulation_steps'],
    early_stopping_patience=3,
)

print(f"\nStage 2 complete: {MODEL_KEY}")
print(f"Model saved to:   {s2_dir}")
f1 = results2.get('test_report', {}).get('macro avg', {}).get('f1-score', 0)
print(f"Test F1-macro:    {f1:.4f}")
