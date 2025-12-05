"""
Classification Agent.

This agent is responsible for:
1. Detecting file types and categories
2. Classifying document content types (invoice, contract, etc.)
3. Identifying key metadata and context
"""

import os
from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.file_type_service import (
    detect_file_type,
    FileTypeInfo,
    FileCategory,
    detect_file_type_tool,
)


# Classification Tools
@tool
def classify_document_content(text_content: str, filename: str) -> str:
    """
    Classify a document based on its text content.
    
    Analyzes the document text to determine:
    - Document type (invoice, contract, report, etc.)
    - Business domain (engineering, finance, legal, etc.)
    - Priority level
    - Key entities mentioned
    
    Args:
        text_content: The extracted text from the document
        filename: Original filename for context
    
    Returns:
        JSON string with classification results
    """
    import json
    import re
    
    text_lower = text_content.lower()
    
    # Document type detection patterns
    doc_types = {
        "invoice": ["invoice", "bill to", "amount due", "payment terms", "subtotal"],
        "contract": ["agreement", "terms and conditions", "hereby agrees", "party of"],
        "purchase_order": ["purchase order", "p.o.", "vendor", "ship to"],
        "report": ["report", "summary", "findings", "analysis", "conclusion"],
        "specification": ["specification", "requirements", "scope of work", "technical"],
        "safety_document": ["safety", "incident", "hazard", "ppe", "emergency"],
        "memo": ["memorandum", "memo", "to:", "from:", "subject:", "re:"],
        "letter": ["dear", "sincerely", "regards", "yours truly"],
        "rfi": ["request for information", "rfi", "clarification"],
        "submittal": ["submittal", "shop drawing", "product data"],
    }
    
    detected_types = []
    for doc_type, patterns in doc_types.items():
        matches = sum(1 for p in patterns if p in text_lower)
        if matches >= 2:
            detected_types.append({"type": doc_type, "confidence": min(matches * 0.2, 1.0)})
    
    # Sort by confidence
    detected_types.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Priority detection
    priority = "normal"
    urgent_patterns = ["urgent", "asap", "immediate", "critical", "emergency", "deadline"]
    if any(p in text_lower for p in urgent_patterns):
        priority = "high"
    
    # Extract key numbers/amounts
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text_content)
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text_content)
    
    result = {
        "filename": filename,
        "detected_document_types": detected_types[:3] if detected_types else [{"type": "general", "confidence": 0.5}],
        "primary_type": detected_types[0]["type"] if detected_types else "general",
        "priority": priority,
        "key_amounts": amounts[:5] if amounts else [],
        "key_dates": dates[:5] if dates else [],
        "word_count": len(text_content.split()),
        "has_financial_data": bool(amounts),
        "has_dates": bool(dates),
    }
    
    return json.dumps(result, indent=2)


@tool
def identify_document_language(text_content: str) -> str:
    """
    Identify the primary language of a document.
    
    Args:
        text_content: The text content to analyze
    
    Returns:
        JSON string with language detection results
    """
    import json
    
    # Simple language detection based on common words
    # In production, would use langdetect or similar library
    english_words = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with"}
    spanish_words = {"el", "la", "de", "en", "que", "y", "los", "del", "es", "un"}
    
    words = set(text_content.lower().split())
    
    english_count = len(words & english_words)
    spanish_count = len(words & spanish_words)
    
    if english_count > spanish_count:
        language = "english"
        confidence = min(english_count / 10, 1.0)
    elif spanish_count > english_count:
        language = "spanish"
        confidence = min(spanish_count / 10, 1.0)
    else:
        language = "unknown"
        confidence = 0.3
    
    return json.dumps({
        "language": language,
        "confidence": confidence,
        "english_indicator_count": english_count,
        "spanish_indicator_count": spanish_count,
    })


# Classification Agent Tools
CLASSIFICATION_TOOLS = [
    detect_file_type_tool,
    classify_document_content,
    identify_document_language,
]


def create_classification_agent(model=None):
    """
    Create the classification agent.
    
    Args:
        model: Optional LLM model. If not provided, uses Gemini with key rotation.
    
    Returns:
        A LangGraph agent for document classification
    """
    if model is None:
        from services.llm_provider import get_model
        model = get_model(model_name="gemini-2.5-flash-lite")
    
    agent = create_react_agent(
        model=model,
        tools=CLASSIFICATION_TOOLS,
        name="classification_agent",
        prompt="""You are a document classification specialist. Your job is to:

1. Analyze files to determine their type (PDF, CAD, spreadsheet, etc.)
2. Classify document content into categories (invoice, contract, report, etc.)
3. Identify key metadata like language, priority, and important entities
4. Provide confidence scores for your classifications

When analyzing a document:
- First detect the file type using the file type tool
- Then classify the content if text is available
- Report your findings in a structured format

Be precise and thorough in your analysis. If you're uncertain, indicate low confidence."""
    )
    
    return agent
