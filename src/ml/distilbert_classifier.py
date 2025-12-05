"""
Multilabel Document Classifier.

This module provides a privacy-safe document classification system using
a custom-trained multilabel model with TF-IDF vectorization. The classifier
ONLY processes email body/message text - never the document content itself.

Key Features:
- Loads custom multilabel model for document categorization
- Classifies based on email subject + body text only
- Returns multiple categories with confidence scores (multilabel)
- Fallback to rule-based classification when model unavailable

Model Artifacts Required:
- multilabel_model.pkl: The trained classifier
- multilabel_vectorizer.pkl: TF-IDF vectorizer
- multilabel_binarizer.pkl: Label binarizer for multilabel output
"""

import os
import re
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Model configuration
# Get the project root (parent of src directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / "models")
_ENV_MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
MODEL_PATH = _ENV_MODEL_PATH if _ENV_MODEL_PATH else _DEFAULT_MODEL_PATH
CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.6"))

# Joblib for loading pkl files
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class DocumentCategory(str, Enum):
    """Document categories matching model output labels."""
    COMPLIANCE = "compliance"
    ENGINEER = "engineer"
    FINANCE = "finance"
    HR = "hr"
    LEGAL = "legal"
    OPERATIONS = "operations"
    PROCUREMENT = "procurement"
    PROJECT_LEADER = "project_leader"
    SAFETY = "safety"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result from document classification."""
    categories: List[DocumentCategory]  # Multiple categories for multilabel
    confidence: float  # Overall confidence (average or max)
    all_scores: Dict[str, float]
    model_used: str  # "multilabel" or "rule-based"
    requires_review: bool = False
    
    @property
    def category(self) -> DocumentCategory:
        """Primary category (first predicted or UNKNOWN)."""
        return self.categories[0] if self.categories else DocumentCategory.UNKNOWN
    
    @property
    def category_names(self) -> List[str]:
        """List of category names as strings."""
        return [cat.value for cat in self.categories]


# ============================================================
# Rule-Based Fallback Classifier
# ============================================================

# Keywords for rule-based classification (fallback)
# Matches model output: ['Compliance', 'Engineer', 'Finance', 'HR', 'Legal', 'Operations', 'Procurement', 'Project Leader', 'Safety']
CATEGORY_KEYWORDS: Dict[DocumentCategory, List[str]] = {
    DocumentCategory.ENGINEER: [
        "engineering", "technical", "design", "cad", "bim", "specification",
        "drawing", "blueprint", "schematic", "mechanical", "electrical",
        "structural", "civil", "architecture", "construction", "rfi",
        "submittal", "shop drawing", "as-built", "p&id", "isometric"
    ],
    DocumentCategory.FINANCE: [
        "invoice", "payment", "budget", "cost", "expense", "financial",
        "billing", "receipt", "purchase order", "po", "quote", "estimate",
        "pricing", "revenue", "profit", "loss", "tax", "audit", "accounting"
    ],
    DocumentCategory.PROCUREMENT: [
        "procurement", "vendor", "supplier", "rfp", "rfq", "bid",
        "contract", "agreement", "purchase", "supply", "material",
        "inventory", "order", "delivery", "logistics", "sourcing"
    ],
    DocumentCategory.HR: [
        "employee", "hr", "human resources", "personnel", "hiring",
        "recruitment", "training", "policy", "handbook", "benefits",
        "payroll", "leave", "performance", "onboarding", "termination",
        "salary", "medical", "health insurance"
    ],
    DocumentCategory.OPERATIONS: [
        "operations", "maintenance", "field", "site", "inspection",
        "schedule", "work order", "task", "procedure", "sop",
        "standard operating", "daily report", "progress", "status"
    ],
    DocumentCategory.SAFETY: [
        "safety", "incident", "accident", "hazard", "risk", "osha",
        "ppe", "emergency", "evacuation", "first aid", "injury",
        "near miss", "jsa", "job safety analysis", "toolbox talk"
    ],
    DocumentCategory.COMPLIANCE: [
        "compliance", "regulation", "permit", "license", "certification",
        "audit", "inspection", "standard", "code", "requirement",
        "epa", "environmental", "regulatory", "violation", "corrective"
    ],
    DocumentCategory.LEGAL: [
        "legal", "law", "attorney", "contract", "agreement", "lawsuit",
        "litigation", "dispute", "claim", "liability", "indemnity",
        "terms", "conditions", "nda", "confidential", "court"
    ],
    DocumentCategory.PROJECT_LEADER: [
        "project leader", "project manager", "pm", "leadership",
        "milestone", "timeline", "gantt", "project plan", "kickoff",
        "executive", "management", "strategy", "summary", "report",
        "dashboard", "kpi", "metrics", "quarterly", "annual", "board"
    ],
}

# High-confidence phrase patterns (exact matches get higher scores)
PHRASE_PATTERNS: Dict[DocumentCategory, List[str]] = {
    DocumentCategory.ENGINEER: [
        "technical specification", "engineering drawing", "shop drawing",
        "as-built drawing", "rfi response", "design review"
    ],
    DocumentCategory.FINANCE: [
        "purchase order", "payment request", "invoice attached",
        "budget report", "cost estimate", "financial statement"
    ],
    DocumentCategory.PROCUREMENT: [
        "request for proposal", "request for quote", "vendor quote",
        "bid submission", "supply agreement", "material order"
    ],
    DocumentCategory.HR: [
        "job application", "employee handbook", "performance review",
        "leave request", "new hire", "exit interview", "salary revision",
        "medical documents", "health insurance"
    ],
    DocumentCategory.OPERATIONS: [
        "daily report", "work order", "maintenance request",
        "site inspection", "progress report", "field report"
    ],
    DocumentCategory.SAFETY: [
        "incident report", "safety inspection", "near miss",
        "job safety analysis", "safety meeting", "toolbox talk"
    ],
    DocumentCategory.COMPLIANCE: [
        "audit report", "compliance review", "permit application",
        "regulatory filing", "inspection report", "certification"
    ],
    DocumentCategory.LEGAL: [
        "legal notice", "contract review", "nda agreement",
        "terms and conditions", "legal opinion", "court filing"
    ],
    DocumentCategory.PROJECT_LEADER: [
        "project status", "executive summary", "board meeting",
        "quarterly review", "strategic plan", "project update",
        "project leader", "project manager"
    ],
}


def rule_based_classify(text: str) -> ClassificationResult:
    """
    Rule-based classification using keyword and phrase matching.
    
    Used as fallback when ML model is not available.
    Returns multiple categories (multilabel support).
    """
    text_lower = text.lower()
    scores: Dict[str, float] = {}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0.0
        matches = 0
        
        # Check phrase patterns first (higher weight)
        if category in PHRASE_PATTERNS:
            for phrase in PHRASE_PATTERNS[category]:
                if phrase in text_lower:
                    score += 5.0  # High boost for phrase matches
                    matches += 1
        
        # Then check individual keywords
        for keyword in keywords:
            if keyword in text_lower:
                # Boost for exact word matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    score += 2.0
                else:
                    score += 1.0
                matches += 1
        
        # Normalize score (adjusted for phrase patterns)
        max_possible = (len(keywords) * 2.0) + (len(PHRASE_PATTERNS.get(category, [])) * 5.0)
        if max_possible > 0:
            scores[category.value] = min(score / (max_possible * 0.15), 1.0)
        else:
            scores[category.value] = 0.0
    
    # Find all categories above threshold (multilabel)
    matched_categories = []
    for cat_value, score in scores.items():
        if score >= CONFIDENCE_THRESHOLD:
            matched_categories.append(DocumentCategory(cat_value))
    
    # Sort by score descending
    matched_categories.sort(key=lambda c: scores[c.value], reverse=True)
    
    if matched_categories:
        avg_confidence = sum(scores[c.value] for c in matched_categories) / len(matched_categories)
        return ClassificationResult(
            categories=matched_categories,
            confidence=avg_confidence,
            all_scores=scores,
            model_used="rule-based",
            requires_review=avg_confidence < 0.8
        )
    
    # No confident match - needs human review
    return ClassificationResult(
        categories=[DocumentCategory.UNKNOWN],
        confidence=max(scores.values()) if scores else 0.0,
        all_scores=scores,
        model_used="rule-based",
        requires_review=True
    )


# ============================================================
# Multilabel Classifier
# ============================================================

# Mapping from model output labels to DocumentCategory
# Model outputs: ['Compliance', 'Engineer', 'Finance', 'HR', 'Legal', 'Operations', 'Procurement', 'Project Leader', 'Safety']
LABEL_TO_CATEGORY: Dict[str, DocumentCategory] = {
    "compliance": DocumentCategory.COMPLIANCE,
    "engineer": DocumentCategory.ENGINEER,
    "finance": DocumentCategory.FINANCE,
    "hr": DocumentCategory.HR,
    "legal": DocumentCategory.LEGAL,
    "operations": DocumentCategory.OPERATIONS,
    "procurement": DocumentCategory.PROCUREMENT,
    "project leader": DocumentCategory.PROJECT_LEADER,
    "project_leader": DocumentCategory.PROJECT_LEADER,
    "safety": DocumentCategory.SAFETY,
}


class MultilabelClassifier:
    """
    Multilabel document classifier using TF-IDF vectorization.
    
    Loads a custom-trained model for multilabel classification.
    The model predicts multiple categories for each input text.
    
    Model Artifacts:
    - multilabel_model.pkl: The trained classifier
    - multilabel_vectorizer.pkl: TF-IDF vectorizer
    - multilabel_binarizer.pkl: Label binarizer for multilabel output
    """
    
    _instance: Optional["MultilabelClassifier"] = None
    _model = None
    _vectorizer = None
    _binarizer = None
    _is_loaded: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self) -> bool:
        """
        Load the multilabel model artifacts from disk.
        
        Returns True if model loaded successfully, False otherwise.
        """
        if self._is_loaded:
            return True
        
        if not JOBLIB_AVAILABLE:
            print("⚠️ joblib library not installed. Install with: pip install joblib")
            print("   Using rule-based classification as fallback.")
            return False
        
        try:
            # Check if model path exists
            if not os.path.exists(MODEL_PATH):
                print(f"⚠️ Model directory not found at {MODEL_PATH}. Using rule-based classification.")
                return False
            
            # Check for required model files
            model_file = os.path.join(MODEL_PATH, "multilabel_model.pkl")
            vectorizer_file = os.path.join(MODEL_PATH, "multilabel_vectorizer.pkl")
            binarizer_file = os.path.join(MODEL_PATH, "multilabel_binarizer.pkl")
            
            if not all(os.path.exists(f) for f in [model_file, vectorizer_file, binarizer_file]):
                print(f"⚠️ Model artifacts not found in {MODEL_PATH}. Required files:")
                print("   - multilabel_model.pkl")
                print("   - multilabel_vectorizer.pkl")
                print("   - multilabel_binarizer.pkl")
                print("   Using rule-based classification as fallback.")
                return False
            
            print(f"📦 Loading multilabel model from {MODEL_PATH}...")
            
            # Load model artifacts
            self._model = joblib.load(model_file)
            self._vectorizer = joblib.load(vectorizer_file)
            self._binarizer = joblib.load(binarizer_file)
            
            self._is_loaded = True
            print(f"✅ Multilabel model loaded successfully")
            print(f"   Classes: {list(self._binarizer.classes_)}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ Falling back to rule-based classification.")
            return False
    
    def _label_to_category(self, label: str) -> DocumentCategory:
        """Convert a model label to DocumentCategory enum."""
        label_lower = label.lower().strip()
        
        # Direct match
        if label_lower in LABEL_TO_CATEGORY:
            return LABEL_TO_CATEGORY[label_lower]
        
        # Try to find partial match
        for key, cat in LABEL_TO_CATEGORY.items():
            if key in label_lower or label_lower in key:
                return cat
        
        # Check if it matches a DocumentCategory value directly
        try:
            return DocumentCategory(label_lower)
        except ValueError:
            pass
        
        return DocumentCategory.UNKNOWN
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text using the multilabel model.
        
        Args:
            text: Email body/message text to classify (NOT document content)
            
        Returns:
            ClassificationResult with multiple categories
        """
        # Try to load model if not already loaded
        if not self._is_loaded:
            if not self.load_model():
                # Fallback to rule-based
                return rule_based_classify(text)
        
        if not text or not text.strip():
            return ClassificationResult(
                categories=[DocumentCategory.UNKNOWN],
                confidence=0.0,
                all_scores={},
                model_used="multilabel",
                requires_review=True
            )
        
        try:
            # Vectorize input
            input_tfidf = self._vectorizer.transform([text])
            
            # Predict - returns binary matrix
            pred_matrix = self._model.predict(input_tfidf)
            
            # Convert binary format back to labels
            pred_labels = self._binarizer.inverse_transform(pred_matrix)
            labels = list(pred_labels[0]) if pred_labels[0] else []
            
            # Get prediction probabilities if available
            all_scores: Dict[str, float] = {}
            if hasattr(self._model, 'predict_proba'):
                try:
                    proba = self._model.predict_proba(input_tfidf)
                    # proba can be a list of arrays (one per class) for multilabel
                    if isinstance(proba, list):
                        for idx, class_proba in enumerate(proba):
                            class_name = self._binarizer.classes_[idx]
                            # Get probability of positive class
                            prob = float(class_proba[0][1]) if class_proba.shape[1] > 1 else float(class_proba[0][0])
                            all_scores[class_name] = prob
                    else:
                        # Single array with all probabilities
                        for idx, prob in enumerate(proba[0]):
                            class_name = self._binarizer.classes_[idx]
                            all_scores[class_name] = float(prob)
                except Exception:
                    # Some classifiers don't support predict_proba well
                    pass
            
            # Convert labels to DocumentCategory
            categories = []
            for label in labels:
                cat = self._label_to_category(str(label))
                if cat not in categories:
                    categories.append(cat)
            
            # If no categories predicted, mark as unknown
            if not categories:
                categories = [DocumentCategory.UNKNOWN]
            
            # Calculate confidence
            if all_scores and labels:
                confidence = sum(all_scores.get(str(l), 0.5) for l in labels) / len(labels)
            else:
                confidence = 0.7 if labels else 0.0  # Default confidence if proba not available
            
            requires_review = confidence < CONFIDENCE_THRESHOLD or DocumentCategory.UNKNOWN in categories
            
            return ClassificationResult(
                categories=categories,
                confidence=confidence,
                all_scores=all_scores,
                model_used="multilabel",
                requires_review=requires_review
            )
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to rule-based
            return rule_based_classify(text)
    
    def predict_raw(self, text: str) -> List[str]:
        """
        Get raw predicted labels without wrapping in ClassificationResult.
        
        Useful for simple integrations that just need the label list.
        """
        if not self._is_loaded:
            if not self.load_model():
                return []
        
        if not text or not text.strip():
            return []
        
        try:
            input_tfidf = self._vectorizer.transform([text])
            pred_matrix = self._model.predict(input_tfidf)
            pred_labels = self._binarizer.inverse_transform(pred_matrix)
            return list(pred_labels[0]) if pred_labels[0] else []
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return []
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_classes(self) -> List[str]:
        """Get list of all possible classes."""
        if self._is_loaded and self._binarizer is not None:
            return list(self._binarizer.classes_)
        return []


# ============================================================
# Global Classifier Instance
# ============================================================

_classifier: Optional[MultilabelClassifier] = None


def get_classifier() -> MultilabelClassifier:
    """Get or create the classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = MultilabelClassifier()
    return _classifier


def classify_document(
    subject: str,
    body: str,
    filename: Optional[str] = None
) -> ClassificationResult:
    """
    Classify a document based on email/message metadata.
    
    PRIVACY SAFE: Only uses subject/body text, never document content.
    
    Args:
        subject: Email subject or message caption
        body: Email body or Telegram message
        filename: Optional filename for additional context
        
    Returns:
        ClassificationResult with categories and confidence (multilabel)
    """
    # Combine available text for classification
    text_parts = []
    
    if subject:
        text_parts.append(f"Subject: {subject}")
    
    if body:
        text_parts.append(body)
    
    if filename:
        # Add filename as hint (but don't rely on it alone)
        text_parts.append(f"Attachment: {filename}")
    
    combined_text = "\n".join(text_parts)
    
    if not combined_text.strip():
        # No text to classify
        return ClassificationResult(
            categories=[DocumentCategory.UNKNOWN],
            confidence=0.0,
            all_scores={},
            model_used="none",
            requires_review=True
        )
    
    classifier = get_classifier()
    return classifier.classify(combined_text)


def classify_text(text: str) -> List[str]:
    """
    Simple classification function that returns raw label list.
    
    Args:
        text: Text to classify
        
    Returns:
        List of predicted category labels
    """
    classifier = get_classifier()
    return classifier.predict_raw(text)


# ============================================================
# Model Management
# ============================================================

def preload_model() -> bool:
    """
    Preload the multilabel model at startup.
    
    Call this during application initialization to avoid
    cold start delays on first classification request.
    """
    classifier = get_classifier()
    return classifier.load_model()


def get_model_info() -> Dict[str, any]:
    """Get information about the loaded model."""
    classifier = get_classifier()
    
    return {
        "model_path": MODEL_PATH,
        "is_loaded": classifier.is_loaded(),
        "model_type": "multilabel",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "categories": [c.value for c in DocumentCategory if c != DocumentCategory.UNKNOWN],
        "classes": classifier.get_classes() if classifier.is_loaded() else [],
        "fallback": "rule-based" if not classifier.is_loaded() else None
    }
