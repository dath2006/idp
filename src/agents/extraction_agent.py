"""
Extraction Agent.

This agent is responsible for:
1. Extracting structured data from documents
2. Converting unstructured text into JSON format
3. Identifying key fields based on document type
"""

import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Extraction schemas for different document types
EXTRACTION_SCHEMAS = {
    "invoice": {
        "invoice_number": "string",
        "invoice_date": "date",
        "due_date": "date",
        "vendor_name": "string",
        "vendor_address": "string",
        "bill_to": "string",
        "line_items": [{"description": "string", "quantity": "number", "unit_price": "number", "amount": "number"}],
        "subtotal": "number",
        "tax": "number",
        "total": "number",
        "payment_terms": "string",
    },
    "contract": {
        "contract_title": "string",
        "parties": ["string"],
        "effective_date": "date",
        "expiration_date": "date",
        "contract_value": "number",
        "key_terms": ["string"],
        "obligations": ["string"],
        "termination_conditions": ["string"],
    },
    "purchase_order": {
        "po_number": "string",
        "order_date": "date",
        "delivery_date": "date",
        "vendor": "string",
        "ship_to": "string",
        "line_items": [{"item": "string", "quantity": "number", "unit_price": "number"}],
        "total": "number",
        "payment_terms": "string",
    },
    "safety_report": {
        "report_type": "string",
        "incident_date": "date",
        "location": "string",
        "description": "string",
        "personnel_involved": ["string"],
        "injuries": "string",
        "root_cause": "string",
        "corrective_actions": ["string"],
        "severity": "string",
    },
    "technical_specification": {
        "document_title": "string",
        "project_name": "string",
        "revision": "string",
        "date": "date",
        "scope": "string",
        "requirements": ["string"],
        "specifications": {"key": "value"},
        "references": ["string"],
    },
    "rfi": {
        "rfi_number": "string",
        "date": "date",
        "project": "string",
        "from": "string",
        "to": "string",
        "subject": "string",
        "question": "string",
        "response_needed_by": "date",
    },
    "general": {
        "title": "string",
        "date": "date",
        "author": "string",
        "summary": "string",
        "key_points": ["string"],
    },
}


@tool
def extract_invoice_data(text_content: str) -> str:
    """
    Extract structured data from an invoice document.
    
    Args:
        text_content: The text content of the invoice
    
    Returns:
        JSON string with extracted invoice fields
    """
    import re
    
    data = {
        "invoice_number": None,
        "invoice_date": None,
        "vendor_name": None,
        "total": None,
        "line_items": [],
        "payment_terms": None,
    }
    
    # Extract invoice number
    inv_match = re.search(r'invoice\s*(?:number|no|#)?[:\s]*([A-Z0-9-]+)', text_content, re.IGNORECASE)
    if inv_match:
        data["invoice_number"] = inv_match.group(1)
    
    # Extract dates
    date_matches = re.findall(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', text_content)
    if date_matches:
        data["invoice_date"] = date_matches[0]
    
    # Extract total
    total_match = re.search(r'total[:\s]*\$?([\d,]+\.?\d*)', text_content, re.IGNORECASE)
    if total_match:
        data["total"] = float(total_match.group(1).replace(',', ''))
    
    # Extract payment terms
    terms_match = re.search(r'(?:payment\s*terms?|net\s*\d+)[:\s]*([^\n]+)', text_content, re.IGNORECASE)
    if terms_match:
        data["payment_terms"] = terms_match.group(1).strip()
    
    return json.dumps(data, indent=2)


@tool
def extract_contract_data(text_content: str) -> str:
    """
    Extract structured data from a contract document.
    
    Args:
        text_content: The text content of the contract
    
    Returns:
        JSON string with extracted contract fields
    """
    import re
    
    data = {
        "contract_title": None,
        "parties": [],
        "effective_date": None,
        "contract_value": None,
        "key_terms": [],
    }
    
    # Try to find title (usually first line or after "Agreement")
    title_match = re.search(r'^(.+(?:agreement|contract))', text_content, re.IGNORECASE | re.MULTILINE)
    if title_match:
        data["contract_title"] = title_match.group(1).strip()
    
    # Find parties
    party_matches = re.findall(r'(?:between|party)[:\s]+([^,\n]+)', text_content, re.IGNORECASE)
    data["parties"] = [p.strip() for p in party_matches[:2]]
    
    # Find effective date
    date_match = re.search(r'effective\s*(?:date)?[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text_content, re.IGNORECASE)
    if date_match:
        data["effective_date"] = date_match.group(1)
    
    # Find contract value
    value_match = re.search(r'(?:contract\s*value|amount|sum)[:\s]*\$?([\d,]+\.?\d*)', text_content, re.IGNORECASE)
    if value_match:
        data["contract_value"] = float(value_match.group(1).replace(',', ''))
    
    return json.dumps(data, indent=2)


@tool
def extract_general_metadata(text_content: str, filename: str) -> str:
    """
    Extract general metadata from any document.
    
    Args:
        text_content: The text content of the document
        filename: The original filename
    
    Returns:
        JSON string with extracted metadata
    """
    import re
    
    # Basic metadata
    lines = text_content.strip().split('\n')
    
    data = {
        "filename": filename,
        "title": lines[0][:100] if lines else filename,
        "word_count": len(text_content.split()),
        "line_count": len(lines),
        "dates_found": [],
        "amounts_found": [],
        "emails_found": [],
        "phone_numbers_found": [],
    }
    
    # Extract dates
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text_content)
    data["dates_found"] = list(set(dates))[:5]
    
    # Extract monetary amounts
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text_content)
    data["amounts_found"] = list(set(amounts))[:5]
    
    # Extract emails
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text_content)
    data["emails_found"] = list(set(emails))[:5]
    
    # Extract phone numbers
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_content)
    data["phone_numbers_found"] = list(set(phones))[:5]
    
    return json.dumps(data, indent=2)


@tool
def generate_document_summary(text_content: str, max_length: int = 500) -> str:
    """
    Generate a summary of the document content.
    
    This creates an extractive summary by identifying key sentences.
    
    Args:
        text_content: The text content to summarize
        max_length: Maximum length of the summary in characters
    
    Returns:
        A concise summary of the document
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text_content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "Document content too short to summarize."
    
    # Simple extractive summary - take first few sentences
    summary_sentences = []
    current_length = 0
    
    for sentence in sentences[:10]:  # Look at first 10 sentences
        if current_length + len(sentence) < max_length:
            summary_sentences.append(sentence)
            current_length += len(sentence)
        else:
            break
    
    if summary_sentences:
        return ". ".join(summary_sentences) + "."
    else:
        return sentences[0][:max_length] + "..."


@tool
def get_extraction_schema(document_type: str) -> str:
    """
    Get the extraction schema for a specific document type.
    
    Args:
        document_type: The type of document (invoice, contract, etc.)
    
    Returns:
        JSON schema for the document type
    """
    schema = EXTRACTION_SCHEMAS.get(document_type.lower(), EXTRACTION_SCHEMAS["general"])
    return json.dumps({
        "document_type": document_type,
        "schema": schema
    }, indent=2)


# Extraction Agent Tools
EXTRACTION_TOOLS = [
    extract_invoice_data,
    extract_contract_data,
    extract_general_metadata,
    generate_document_summary,
    get_extraction_schema,
]


def create_extraction_agent(model=None):
    """
    Create the extraction agent.
    
    Args:
        model: Optional LLM model. If not provided, uses Gemini with key rotation.
    
    Returns:
        A LangGraph agent for data extraction
    """
    if model is None:
        from services.llm_provider import get_model
        model = get_model(model_name="gemini-2.5-flash-lite")
    
    agent = create_react_agent(
        model=model,
        tools=EXTRACTION_TOOLS,
        name="extraction_agent",
        prompt="""You are a data extraction specialist. Your job is to:

1. Extract structured data from documents based on their type
2. Convert unstructured text into clean JSON format
3. Identify and extract key fields like dates, amounts, names, etc.
4. Generate concise summaries of document content

When extracting data:
- First determine the document type to select the right extraction schema
- Use the appropriate extraction tool for the document type
- Always extract general metadata as a baseline
- Generate a summary for human review

Be thorough but avoid hallucinating data that isn't present in the document.
If a field cannot be found, leave it as null rather than guessing."""
    )
    
    return agent
