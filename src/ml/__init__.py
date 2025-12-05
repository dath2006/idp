"""
Machine Learning module for privacy-first document classification.
"""

from .distilbert_classifier import (
    MultilabelClassifier,
    DocumentCategory,
    ClassificationResult,
    classify_document,
    classify_text,
    get_classifier,
    preload_model,
    get_model_info,
    rule_based_classify,
    LABEL_TO_CATEGORY,
)

__all__ = [
    "MultilabelClassifier",
    "DocumentCategory",
    "ClassificationResult",
    "classify_document",
    "classify_text",
    "get_classifier",
    "preload_model",
    "get_model_info",
    "rule_based_classify",
    "LABEL_TO_CATEGORY",
]
