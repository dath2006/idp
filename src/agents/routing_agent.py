"""
Routing Agent.

This agent is responsible for:
1. Determining which departments should receive a document
2. Assigning priority levels for routing
3. Sending notifications to relevant stakeholders
"""

import os
import json
from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.routing_service import (
    analyze_document_for_routing,
    get_department_contacts,
)
from services.notification_service import send_document_notification
from config.departments import (
    Department,
    DEPARTMENTS,
    get_all_departments,
)


@tool
def list_all_departments() -> str:
    """
    List all available departments with their descriptions.
    
    Returns:
        JSON string with all department information
    """
    departments = get_all_departments()
    result = []
    
    for dept in departments:
        result.append({
            "id": dept.id.value,
            "name": dept.name,
            "description": dept.description,
            "email": dept.email,
            "priority": dept.priority,
        })
    
    return json.dumps(result, indent=2)


@tool
def determine_routing_priority(
    document_type: str,
    keywords_found: str,
    has_deadline: bool = False
) -> str:
    """
    Determine the priority level for routing a document.
    
    Args:
        document_type: Type of document (invoice, contract, safety_report, etc.)
        keywords_found: Comma-separated list of keywords found in document
        has_deadline: Whether the document mentions a deadline
    
    Returns:
        JSON string with priority determination
    """
    # High priority document types
    high_priority_types = {"safety_report", "incident_report", "compliance", "legal"}
    
    # Urgent keywords
    urgent_keywords = {"urgent", "asap", "immediate", "critical", "emergency", "deadline"}
    
    keywords_set = set(k.strip().lower() for k in keywords_found.split(","))
    
    # Determine priority
    if document_type.lower() in high_priority_types:
        priority = "high"
        reason = f"Document type '{document_type}' is classified as high priority"
    elif keywords_set & urgent_keywords:
        priority = "high"
        reason = f"Urgent keywords detected: {keywords_set & urgent_keywords}"
    elif has_deadline:
        priority = "normal"
        reason = "Document has deadline, prioritizing accordingly"
    else:
        priority = "normal"
        reason = "Standard document, normal priority"
    
    return json.dumps({
        "priority": priority,
        "reason": reason,
        "document_type": document_type,
        "urgent_keywords_found": list(keywords_set & urgent_keywords),
    }, indent=2)


@tool
def validate_routing_decision(
    departments: str,
    document_type: str,
    file_extension: str
) -> str:
    """
    Validate that a routing decision makes sense.
    
    Args:
        departments: Comma-separated list of target departments
        document_type: The classified document type
        file_extension: The file extension
    
    Returns:
        JSON string with validation results and any warnings
    """
    dept_list = [d.strip().lower() for d in departments.split(",")]
    
    validation = {
        "valid": True,
        "warnings": [],
        "suggestions": [],
    }
    
    # Check for unknown departments
    valid_depts = {d.value for d in Department}
    unknown = set(dept_list) - valid_depts
    if unknown:
        validation["valid"] = False
        validation["warnings"].append(f"Unknown departments: {unknown}")
    
    # Check for expected routing based on file type
    expected_routing = {
        ".dwg": ["engineering"],
        ".dxf": ["engineering"],
        ".ifc": ["engineering"],
        ".xlsx": ["finance", "operations"],
        ".pptx": ["management"],
    }
    
    if file_extension.lower() in expected_routing:
        expected = set(expected_routing[file_extension.lower()])
        if not (expected & set(dept_list)):
            validation["warnings"].append(
                f"File type {file_extension} typically routes to {expected}, but routing to {dept_list}"
            )
    
    # Check for document type routing
    type_routing = {
        "invoice": ["finance"],
        "contract": ["legal", "procurement"],
        "safety_report": ["safety", "compliance"],
        "technical_specification": ["engineering"],
    }
    
    if document_type.lower() in type_routing:
        expected = set(type_routing[document_type.lower()])
        if not (expected & set(dept_list)):
            validation["suggestions"].append(
                f"Consider adding {expected} for {document_type} documents"
            )
    
    return json.dumps(validation, indent=2)


@tool
def create_routing_summary(
    filename: str,
    document_type: str,
    departments: str,
    summary: str,
    extracted_data: str
) -> str:
    """
    Create a human-readable routing summary for notification.
    
    Args:
        filename: Name of the document
        document_type: Classified document type
        departments: Comma-separated list of target departments
        summary: Document summary
        extracted_data: JSON string of extracted data
    
    Returns:
        Formatted routing summary
    """
    dept_list = [d.strip() for d in departments.split(",")]
    
    try:
        data = json.loads(extracted_data)
    except:
        data = {}
    
    lines = [
        "=" * 50,
        "DOCUMENT ROUTING SUMMARY",
        "=" * 50,
        f"",
        f"📄 Document: {filename}",
        f"📋 Type: {document_type}",
        f"🏢 Routed to: {', '.join(dept_list)}",
        f"",
        "📝 Summary:",
        summary,
        f"",
    ]
    
    if data:
        lines.append("📊 Key Extracted Data:")
        for key, value in list(data.items())[:5]:
            if value:
                lines.append(f"  • {key}: {value}")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


# Routing Agent Tools
ROUTING_TOOLS = [
    analyze_document_for_routing,
    get_department_contacts,
    list_all_departments,
    determine_routing_priority,
    validate_routing_decision,
    create_routing_summary,
    send_document_notification,
]


def create_routing_agent(model=None):
    """
    Create the routing agent.
    
    Args:
        model: Optional LLM model. If not provided, uses Gemini with key rotation.
    
    Returns:
        A LangGraph agent for document routing
    """
    if model is None:
        from services.llm_provider import get_model
        model = get_model(model_name="gemini-2.5-flash-lite")
    
    agent = create_react_agent(
        model=model,
        tools=ROUTING_TOOLS,
        name="routing_agent",
        prompt="""You are a document routing specialist. Your job is to:

1. Determine which departments should receive a document based on its content and type
2. Assign appropriate priority levels for urgent or important documents
3. Validate routing decisions to ensure they make sense
4. Send notifications to the relevant departments

When routing a document:
- First analyze the document content to identify relevant departments
- Consider both file type and content for routing decisions
- Determine priority based on document type and keywords
- Validate your routing decision before finalizing
- Create a clear routing summary for stakeholders
- Send notifications to the target departments

Be thorough in your analysis and err on the side of including departments
that might be interested rather than excluding them."""
    )
    
    return agent
