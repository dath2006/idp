"""
Document Routing Service.

This service provides intelligent routing of documents to appropriate departments
based on file type, content analysis, and keyword matching.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from langchain_core.tools import tool

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.departments import (
    Department,
    DepartmentConfig,
    DEPARTMENTS,
    get_department_for_file_type,
    get_department_keywords,
)
from services.file_type_service import FileTypeInfo, FileCategory


class RoutingDecision(BaseModel):
    """Result of routing analysis."""
    primary_departments: List[Department] = Field(
        description="Primary departments to route document to"
    )
    secondary_departments: List[Department] = Field(
        default_factory=list,
        description="Secondary departments that may be interested"
    )
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence score for each department (0-1)"
    )
    routing_reason: str = Field(
        description="Explanation for the routing decision"
    )
    requires_review: bool = Field(
        default=False,
        description="Whether manual review is recommended"
    )


class ContentAnalysisResult(BaseModel):
    """Result of document content analysis for routing."""
    detected_keywords: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Keywords found per department"
    )
    document_type_hints: List[str] = Field(
        default_factory=list,
        description="Detected document type hints (invoice, contract, etc.)"
    )
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted entities (dates, amounts, names, etc.)"
    )


# Document type patterns for content-based detection
DOCUMENT_TYPE_PATTERNS = {
    "invoice": [
        r"invoice\s*(number|no|#)",
        r"bill\s*to",
        r"amount\s*due",
        r"payment\s*terms",
        r"total\s*:?\s*\$?\d+",
        r"subtotal",
        r"tax\s*:?\s*\$?\d+",
    ],
    "contract": [
        r"agreement\s*between",
        r"terms\s*and\s*conditions",
        r"party\s*(of|to)\s*the\s*first\s*part",
        r"hereby\s*agrees",
        r"effective\s*date",
        r"termination",
        r"indemnification",
    ],
    "purchase_order": [
        r"purchase\s*order",
        r"p\.?o\.?\s*(number|no|#)",
        r"vendor\s*(code|id|number)",
        r"ship\s*to",
        r"delivery\s*date",
    ],
    "technical_spec": [
        r"specifications?",
        r"requirements?",
        r"scope\s*of\s*work",
        r"technical\s*requirements",
        r"design\s*criteria",
        r"performance\s*criteria",
    ],
    "safety_report": [
        r"incident\s*report",
        r"accident\s*report",
        r"near\s*miss",
        r"safety\s*observation",
        r"hazard\s*identification",
        r"root\s*cause\s*analysis",
    ],
    "rfi": [
        r"request\s*for\s*information",
        r"rfi\s*(number|no|#)",
        r"clarification\s*request",
    ],
    "submittal": [
        r"submittal",
        r"shop\s*drawing",
        r"product\s*data",
        r"sample\s*submission",
    ],
}


def analyze_content_for_routing(
    text_content: str,
    file_info: Optional[FileTypeInfo] = None
) -> ContentAnalysisResult:
    """
    Analyze document content to determine routing.
    
    Args:
        text_content: Extracted text from document
        file_info: Optional file type information
    
    Returns:
        ContentAnalysisResult with detected keywords and document types
    """
    text_lower = text_content.lower()
    
    # Get department keywords
    dept_keywords = get_department_keywords()
    
    # Find keywords per department
    detected_keywords: Dict[str, List[str]] = {}
    for dept, keywords in dept_keywords.items():
        found = [kw for kw in keywords if kw.lower() in text_lower]
        if found:
            detected_keywords[dept.value] = found
    
    # Detect document type
    document_type_hints = []
    for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                if doc_type not in document_type_hints:
                    document_type_hints.append(doc_type)
                break
    
    # Extract basic entities (simplified - would use NER in production)
    entities: Dict[str, List[str]] = {}
    
    # Find monetary amounts
    amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text_content)
    if amounts:
        entities["monetary_amounts"] = amounts[:5]  # Limit to first 5
    
    # Find dates
    dates = re.findall(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        text_content,
        re.IGNORECASE
    )
    if dates:
        entities["dates"] = dates[:5]
    
    return ContentAnalysisResult(
        detected_keywords=detected_keywords,
        document_type_hints=document_type_hints,
        entities=entities
    )


def route_by_file_type(file_info: FileTypeInfo) -> RoutingDecision:
    """
    Route document based purely on file type.
    
    Args:
        file_info: File type information
    
    Returns:
        RoutingDecision based on file extension
    """
    departments = get_department_for_file_type(file_info.extension)
    
    if departments:
        confidence = {dept.value: 0.9 for dept in departments}
        return RoutingDecision(
            primary_departments=departments,
            confidence_scores=confidence,
            routing_reason=f"File type {file_info.extension} is associated with {', '.join(d.value for d in departments)}",
            requires_review=False
        )
    else:
        return RoutingDecision(
            primary_departments=[],
            confidence_scores={},
            routing_reason=f"File type {file_info.extension} requires content analysis for routing",
            requires_review=True
        )


def route_by_content(
    content_analysis: ContentAnalysisResult,
    file_info: Optional[FileTypeInfo] = None
) -> RoutingDecision:
    """
    Route document based on content analysis.
    
    Args:
        content_analysis: Result of content analysis
        file_info: Optional file type information
    
    Returns:
        RoutingDecision based on content
    """
    # Calculate scores based on keyword matches
    scores: Dict[str, float] = {}
    
    for dept_value, keywords in content_analysis.detected_keywords.items():
        # Base score from keyword count
        keyword_score = min(len(keywords) * 0.15, 0.6)
        
        # Boost for document type matches
        type_boost = 0.0
        for doc_type in content_analysis.document_type_hints:
            if _document_type_matches_department(doc_type, dept_value):
                type_boost = 0.3
                break
        
        scores[dept_value] = min(keyword_score + type_boost, 1.0)
    
    # Sort by score
    sorted_depts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Primary departments (score >= 0.3)
    primary = [Department(d) for d, s in sorted_depts if s >= 0.3]
    
    # Secondary departments (0.15 <= score < 0.3)
    secondary = [Department(d) for d, s in sorted_depts if 0.15 <= s < 0.3]
    
    # Determine if review is needed
    requires_review = len(primary) == 0 or (len(primary) > 0 and max(scores.values()) < 0.5)
    
    if primary:
        reason = f"Content analysis matched keywords for {', '.join(d.value for d in primary)}"
        if content_analysis.document_type_hints:
            reason += f". Document type detected: {', '.join(content_analysis.document_type_hints)}"
    else:
        reason = "No strong department match found in content. Manual review recommended."
    
    return RoutingDecision(
        primary_departments=primary,
        secondary_departments=secondary,
        confidence_scores=scores,
        routing_reason=reason,
        requires_review=requires_review
    )


def _document_type_matches_department(doc_type: str, department: str) -> bool:
    """Check if document type is associated with a department."""
    type_to_dept = {
        "invoice": ["finance"],
        "contract": ["legal", "procurement"],
        "purchase_order": ["procurement", "finance"],
        "technical_spec": ["engineering"],
        "safety_report": ["safety", "compliance"],
        "rfi": ["engineering"],
        "submittal": ["engineering"],
    }
    return department in type_to_dept.get(doc_type, [])


def get_department_emails(departments: List[Department]) -> List[str]:
    """Get email addresses for a list of departments."""
    emails = []
    for dept in departments:
        config = DEPARTMENTS.get(dept)
        if config:
            emails.append(config.email)
    return emails


# Combined routing function
async def route_document(
    filename: str,
    text_content: Optional[str] = None,
    file_info: Optional[FileTypeInfo] = None
) -> Tuple[RoutingDecision, List[str]]:
    """
    Route a document to appropriate departments.
    
    Args:
        filename: Name of the file
        text_content: Optional extracted text content
        file_info: Optional file type information
    
    Returns:
        Tuple of (RoutingDecision, list of department emails)
    """
    from services.file_type_service import detect_file_type
    
    # Get file type info if not provided
    if file_info is None:
        file_info = detect_file_type(filename)
    
    # First try file-type-based routing
    file_type_decision = route_by_file_type(file_info)
    
    if file_type_decision.primary_departments and not file_type_decision.requires_review:
        # File type gives clear routing
        emails = get_department_emails(file_type_decision.primary_departments)
        return file_type_decision, emails
    
    # Need content-based routing
    if text_content:
        content_analysis = analyze_content_for_routing(text_content, file_info)
        content_decision = route_by_content(content_analysis, file_info)
        
        # Merge with file type decision if applicable
        if file_type_decision.primary_departments:
            # Combine both results
            all_primary = list(set(file_type_decision.primary_departments + content_decision.primary_departments))
            content_decision.primary_departments = all_primary
        
        emails = get_department_emails(content_decision.primary_departments)
        return content_decision, emails
    
    # No content available, use file type decision or default
    if file_type_decision.primary_departments:
        emails = get_department_emails(file_type_decision.primary_departments)
        return file_type_decision, emails
    
    # Fallback to operations
    default_decision = RoutingDecision(
        primary_departments=[Department.OPERATIONS],
        routing_reason="No specific routing determined, defaulting to Operations for review",
        requires_review=True
    )
    emails = get_department_emails([Department.OPERATIONS])
    return default_decision, emails


# LangChain tools for routing
@tool
def analyze_document_for_routing(text_content: str, filename: str) -> str:
    """
    Analyze document content to determine which departments should receive it.
    
    Args:
        text_content: The text content extracted from the document
        filename: The name of the file being analyzed
    
    Returns:
        A JSON string with routing analysis including detected keywords,
        document type hints, and department recommendations.
    """
    from services.file_type_service import detect_file_type
    
    file_info = detect_file_type(filename)
    content_analysis = analyze_content_for_routing(text_content, file_info)
    routing = route_by_content(content_analysis, file_info)
    
    result = {
        "filename": filename,
        "file_type": file_info.model_dump(),
        "content_analysis": content_analysis.model_dump(),
        "routing_decision": routing.model_dump()
    }
    
    import json
    return json.dumps(result, indent=2, default=str)


@tool
def get_department_contacts(department_names: str) -> str:
    """
    Get contact information for specified departments.
    
    Args:
        department_names: Comma-separated list of department names
    
    Returns:
        A JSON string with department contact information.
    """
    import json
    
    dept_list = [d.strip().lower() for d in department_names.split(",")]
    contacts = {}
    
    for dept_name in dept_list:
        try:
            dept = Department(dept_name)
            config = DEPARTMENTS.get(dept)
            if config:
                contacts[dept_name] = {
                    "name": config.name,
                    "email": config.email,
                    "description": config.description
                }
        except ValueError:
            contacts[dept_name] = {"error": f"Unknown department: {dept_name}"}
    
    return json.dumps(contacts, indent=2)
