"""
BERT-based multi-label intent classifier.

Part of PhD Thesis: Triple-Hybrid RAG with PPO-based L-DWA
Author: Shin Dong-wook <sdw@hoseo.ac.kr>
Thesis Reference: Chapter 3 Section 3 (multi-label rationale),
                  Chapter 5 Section 1 (PPO state — intent_logits 3-dim slot)

Multi-label rationale (thesis Sec 3.3): the rule-based classifier collapses
boundary queries (e.g. "35세 이하 AND 컴공과 교수 중 ICML 논문 저자는?") into
a single "conditional" bucket, losing the multi-hop signal. The BERT head
emits independent sigmoid probabilities for {simple, multi_hop, conditional}
so a query can be both multi_hop AND conditional.

Default backbone: ``klue/bert-base`` (Korean). For non-Korean medical queries
(PubMedQA), pass ``BiomedNLP/PubMedBERT-base-uncased-abstract-fulltext``.

Tokenizer + model are dependency-injected so unit tests can substitute small
fakes; production callers use :meth:`from_pretrained`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

LABELS: tuple[str, str, str] = ("simple", "multi_hop", "conditional")
LABEL_TO_IDX: dict[str, int] = {label: i for i, label in enumerate(LABELS)}


class _LabeledQueryDataset(Dataset):
    def __init__(self, items: Sequence[tuple[str, list[str]]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, list[str]]:
        return self.items[idx]


def _collate(batch: list[tuple[str, list[str]]]) -> tuple[list[str], list[list[str]]]:
    queries = [q for q, _ in batch]
    labels = [lbls for _, lbls in batch]
    return queries, labels


class BertIntentClassifier:
    """Multi-label classifier with BCE loss and sigmoid output.

    Args:
        tokenizer: A HuggingFace-style callable tokenizer (used as
            ``tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=...)``).
        model: A model whose ``forward(input_ids, attention_mask)`` returns an
            object with a ``logits`` attribute of shape ``(batch, 3)``.
        device: PyTorch device string. Default "cpu" (test-friendly).
        max_length: Tokenizer truncation length. Default 128.

    Example:
        >>> clf = BertIntentClassifier.from_pretrained("klue/bert-base")  # doctest: +SKIP
        >>> clf.predict("40세 이하 컴공과 교수 중 ICML 저자는?")  # doctest: +SKIP
        {'simple': 0.04, 'multi_hop': 0.91, 'conditional': 0.95}
    """

    LABELS: tuple[str, str, str] = LABELS

    def __init__(
        self,
        tokenizer: object,
        model: nn.Module,
        device: str = "cpu",
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length
        self.model.to(device)
        self.model.eval()

    # ---------- factories ----------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "klue/bert-base",
        device: str = "cuda",
        max_length: int = 128,
    ) -> "BertIntentClassifier":
        """Load a HuggingFace backbone with a freshly-initialized 3-label head."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(cls.LABELS),
            problem_type="multi_label_classification",
            id2label={i: label for i, label in enumerate(cls.LABELS)},
            label2id={label: i for i, label in enumerate(cls.LABELS)},
        )
        return cls(tokenizer=tokenizer, model=model, device=device, max_length=max_length)

    # ---------- inference ----------

    @torch.no_grad()
    def predict(self, query: str) -> dict[str, float]:
        """Return per-label probabilities for one query."""
        return self.predict_batch([query])[0]

    @torch.no_grad()
    def predict_batch(self, queries: Sequence[str]) -> list[dict[str, float]]:
        """Return per-label probabilities for a batch of queries."""
        if not queries:
            return []
        self.model.eval()
        enc = self.tokenizer(
            list(queries),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits  # (B, 3)
        probs = torch.sigmoid(logits)
        return [
            {label: float(probs[i, j]) for j, label in enumerate(self.LABELS)}
            for i in range(probs.shape[0])
        ]

    @torch.no_grad()
    def predict_logits(self, query: str) -> tuple[float, float, float]:
        """Return the 3-dim BERT logits (pre-sigmoid).

        Used as part of the 18-dim PPO state vector (thesis Eq. 5-1 slot 4–6).
        """
        enc = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[0]
        return (float(logits[0]), float(logits[1]), float(logits[2]))

    # ---------- training ----------

    def train_loop(
        self,
        train_data: Sequence[tuple[str, list[str]]],
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 16,
        weight_decay: float = 0.01,
    ) -> list[float]:
        """Fine-tune on labeled queries. Returns per-epoch mean loss.

        Args:
            train_data: List of ``(query, label_names)`` pairs. ``label_names``
                is a subset of :data:`LABELS` — multi-label is allowed.
            epochs: Number of training epochs.
            lr: Adam learning rate.
            batch_size: Mini-batch size.
            weight_decay: AdamW weight decay.
        """
        if not train_data:
            return []
        loader = DataLoader(
            _LabeledQueryDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        epoch_losses: list[float] = []
        for epoch in range(epochs):
            batch_losses: list[float] = []
            for queries, label_lists in loader:
                target = self._labels_to_tensor(label_lists).to(self.device)
                enc = self.tokenizer(
                    list(queries),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            mean_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(mean_loss)
            logger.info("epoch %d/%d  mean_loss=%.4f", epoch + 1, epochs, mean_loss)
        self.model.eval()
        return epoch_losses

    def _labels_to_tensor(self, label_lists: Iterable[list[str]]) -> torch.Tensor:
        rows: list[list[float]] = []
        for labels in label_lists:
            rows.append([1.0 if label in labels else 0.0 for label in self.LABELS])
        return torch.tensor(rows, dtype=torch.float32)

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        """Save tokenizer + model weights + label list to ``path/`` directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(path))
        self.model.save_pretrained(str(path))
        (path / "labels.json").write_text(json.dumps(list(self.LABELS), ensure_ascii=False))

    @classmethod
    def load(
        cls,
        path: Path,
        device: str = "cuda",
        max_length: int = 128,
    ) -> "BertIntentClassifier":
        """Load a previously saved classifier."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        return cls(tokenizer=tokenizer, model=model, device=device, max_length=max_length)
