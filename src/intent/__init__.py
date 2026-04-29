"""Query intent classification (rule-based and BERT-based)."""

from src.intent.rule_based import QueryIntent, RuleBasedIntent

__all__ = ["RuleBasedIntent", "QueryIntent", "BertIntentClassifier", "LABELS"]


def __getattr__(name):
    if name in ("BertIntentClassifier", "LABELS"):
        from src.intent.bert_classifier import BertIntentClassifier, LABELS  # noqa: F401
        return locals()[name] if name == "LABELS" else BertIntentClassifier
    raise AttributeError(f"module 'src.intent' has no attribute {name!r}")
