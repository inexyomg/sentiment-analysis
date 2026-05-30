import os
import numpy as np
import torch
from typing import List, Dict, Union, Optional

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .data_loader import EKMAN_LABEL_NAMES, EKMAN_ID2LABEL
from .ensemble import soft_voting_proba
from .preprocessor import clean_text


class EmotionClassifier:
    """
    Inference wrapper over one or more fine-tuned emotion models.

    Loads each model from its directory, runs them, and combines probabilities
    via weighted soft-voting (the same math as src.ensemble.soft_voting_proba),
    so the demo, the API and the notebooks all agree.

    Example:
        clf = EmotionClassifier([
            "results/models/rubert",
            "results/models/xlmroberta",
            "results/models/rubert_tiny",
        ])
        clf.predict("Мне очень страшно идти туда одному")
        # [{'fear': 0.71, 'sadness': 0.12, ...}]
    """

    def __init__(
        self,
        model_dirs: Union[str, List[str]],
        weights: Optional[List[float]] = None,
        label_names: Optional[List[str]] = None,
        multi_label: bool = False,
        device: Optional[str] = None,
        clean: bool = True,
    ):
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_label = multi_label
        self.clean = clean
        self.label_names = label_names or EKMAN_LABEL_NAMES
        self.weights = weights
        self.meta_clf = None  # set by from_config() for stacking ensembles

        self.tokenizers, self.models, self.thresholds = [], [], []
        for d in model_dirs:
            self.tokenizers.append(AutoTokenizer.from_pretrained(d))
            model = AutoModelForSequenceClassification.from_pretrained(d).to(self.device).eval()
            self.models.append(model)
            t_path = os.path.join(d, "thresholds.npy")
            self.thresholds.append(np.load(t_path) if os.path.exists(t_path) else None)

        # Derive label names from the first model's config if not given
        if label_names is None and getattr(self.models[0].config, "id2label", None):
            cfg = self.models[0].config.id2label
            if all(not str(v).startswith("LABEL_") for v in cfg.values()):
                self.label_names = [cfg[i] for i in sorted(cfg)]

    @classmethod
    def from_config(
        cls,
        config_dir: str,
        device: Optional[str] = None,
        clean: bool = True,
    ) -> "EmotionClassifier":
        """
        Load an ensemble saved with save_ensemble().

        Example:
            clf = EmotionClassifier.from_config("/kaggle/working/ensemble")
            clf.predict_label(["Мне очень страшно"])
        """
        from .ensemble import load_ensemble_config
        config, meta_clf = load_ensemble_config(config_dir)
        inst = cls(
            model_dirs=config["model_dirs"],
            weights=config.get("weights"),
            label_names=config.get("label_names"),
            device=device,
            clean=clean,
        )
        inst.meta_clf = meta_clf
        return inst

    @torch.no_grad()
    def _model_probs(self, model, tokenizer, texts, max_length, batch_size):
        chunks = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i:i + batch_size], truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            logits = model(**enc).logits
            if self.multi_label:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)
            chunks.append(probs.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def predict_proba(self, texts: Union[str, List[str]],
                      max_length: int = 128, batch_size: int = 32) -> np.ndarray:
        """Return ensemble probability matrix of shape (n_texts, n_classes)."""
        if isinstance(texts, str):
            texts = [texts]
        if self.clean:
            texts = [clean_text(t) for t in texts]

        per_model = [
            self._model_probs(m, tok, texts, max_length, batch_size)
            for m, tok in zip(self.models, self.tokenizers)
        ]

        if self.meta_clf is not None:
            # Stacking: concatenate per-model probs and pass through meta-learner
            X = np.hstack(per_model)
            return self.meta_clf.predict_proba(X)

        return soft_voting_proba(per_model, self.weights)

    def predict(self, texts: Union[str, List[str]],
                top_k: Optional[int] = None,
                max_length: int = 128, batch_size: int = 32) -> List[Dict[str, float]]:
        """Return a ranked {label: probability} dict per input text."""
        proba = self.predict_proba(texts, max_length, batch_size)
        results = []
        for row in proba:
            ranked = sorted(
                ((self.label_names[i], float(p)) for i, p in enumerate(row)),
                key=lambda x: -x[1],
            )
            results.append(dict(ranked[:top_k] if top_k else ranked))
        return results

    def predict_label(self, texts: Union[str, List[str]], **kwargs) -> List[str]:
        """Return only the top emotion label per text."""
        return [max(d, key=d.get) for d in self.predict(texts, top_k=1, **kwargs)]
