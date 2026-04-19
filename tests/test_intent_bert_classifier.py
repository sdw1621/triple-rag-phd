"""Tests for src.intent.bert_classifier.BertIntentClassifier.

Uses a tiny in-memory tokenizer + nn.Module pair to avoid downloading
klue/bert-base (~440 MB). The integration test against a real backbone is
marked @pytest.mark.integration and skipped in normal runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.intent.bert_classifier import LABELS, BertIntentClassifier


# ---------- fakes ----------

class FakeTokenizer:
    """Mimics HF tokenizer __call__ signature; returns torch tensors."""

    SEQ_LEN = 8
    VOCAB_SIZE = 100

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        b = len(texts)
        input_ids = torch.zeros(b, self.SEQ_LEN, dtype=torch.long)
        for i, text in enumerate(texts):
            for j, ch in enumerate(text[: self.SEQ_LEN]):
                input_ids[i, j] = (ord(ch) % (self.VOCAB_SIZE - 1)) + 1
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "fake_tokenizer.txt").write_text("ok")


class FakeBertHead(nn.Module):
    """Token-embedding + mean pool + linear → 3 logits. Trainable end-to-end."""

    def __init__(self, vocab_size: int = FakeTokenizer.VOCAB_SIZE, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.head = nn.Linear(hidden, len(LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.unsqueeze(-1).float()
        emb = self.embed(input_ids) * mask
        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.head(pooled)
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")


@pytest.fixture
def classifier() -> BertIntentClassifier:
    torch.manual_seed(0)
    return BertIntentClassifier(
        tokenizer=FakeTokenizer(),
        model=FakeBertHead(),
        device="cpu",
    )


# ---------- predict ----------

def test_predict_returns_three_label_dict(classifier: BertIntentClassifier) -> None:
    out = classifier.predict("김철수 교수")
    assert set(out.keys()) == set(LABELS)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_predict_batch_returns_one_dict_per_query(classifier: BertIntentClassifier) -> None:
    out = classifier.predict_batch(["q1", "q2", "q3"])
    assert len(out) == 3
    for d in out:
        assert set(d.keys()) == set(LABELS)


def test_predict_batch_empty_returns_empty_list(classifier: BertIntentClassifier) -> None:
    assert classifier.predict_batch([]) == []


def test_predict_logits_returns_three_floats(classifier: BertIntentClassifier) -> None:
    logits = classifier.predict_logits("test")
    assert isinstance(logits, tuple)
    assert len(logits) == 3
    assert all(isinstance(x, float) for x in logits)


# ---------- training ----------

def test_train_loop_returns_per_epoch_losses(classifier: BertIntentClassifier) -> None:
    data = [
        ("김철수 교수는 어느 학과?", ["simple"]),
        ("김철수와 이영희가 함께 한 연구는?", ["multi_hop"]),
        ("40세 이하 컴공 교수는?", ["conditional"]),
        ("35세 이하 AND 컴공 교수의 협력자는?", ["multi_hop", "conditional"]),
    ]
    losses = classifier.train_loop(data, epochs=2, batch_size=2, lr=1e-2)
    assert len(losses) == 2
    assert all(loss > 0 for loss in losses)


def test_train_loop_loss_decreases_on_repeated_data(classifier: BertIntentClassifier) -> None:
    """Overfitting check: loss must drop given enough epochs on small data."""
    data = [
        ("김철수 교수는 컴공?", ["simple"]),
        ("40세 이하 교수는?", ["conditional"]),
    ] * 4
    losses = classifier.train_loop(data, epochs=10, batch_size=2, lr=5e-2)
    assert losses[-1] < losses[0]


def test_train_loop_empty_returns_empty(classifier: BertIntentClassifier) -> None:
    assert classifier.train_loop([], epochs=3) == []


def test_labels_to_tensor_multi_label(classifier: BertIntentClassifier) -> None:
    t = classifier._labels_to_tensor([["simple"], ["multi_hop", "conditional"]])
    assert t.shape == (2, 3)
    assert t[0].tolist() == [1.0, 0.0, 0.0]
    assert t[1].tolist() == [0.0, 1.0, 1.0]


# ---------- save/load (uses real fakes that implement save_pretrained) ----------

def test_save_writes_files(classifier: BertIntentClassifier, tmp_path: Path) -> None:
    target = tmp_path / "ckpt"
    classifier.save(target)
    assert (target / "labels.json").exists()
    assert (target / "fake_tokenizer.txt").exists()
    assert (target / "model.pt").exists()


# ---------- integration (skipped by default) ----------

@pytest.mark.integration
def test_from_pretrained_klue_bert_base() -> None:
    """End-to-end check against real klue/bert-base. ~440 MB download; manual."""
    clf = BertIntentClassifier.from_pretrained("klue/bert-base", device="cpu")
    out = clf.predict("김철수 교수는 어느 학과 소속인가?")
    assert set(out.keys()) == set(LABELS)
