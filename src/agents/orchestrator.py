"""
Multi-Agent Orchestrator for Document Processing.

This module creates a supervisor-based multi-agent workflow that coordinates:
1. Classification Agent - File type and content classification
2. Extraction Agent - Structured data extraction
3. Routing Agent - Department routing and notifications

The supervisor orchestrates these agents to process documents end-to-end.
"""

import os
import json
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.classification_agent import create_classification_agent, CLASSIFICATION_TOOLS
from agents.extraction_agent import create_extraction_agent, EXTRACTION_TOOLS
from agents.routing_agent import create_routing_agent, ROUTING_TOOLS
from services.document_service import DOCUMENT_TOOLS


class DocumentProcessingState(TypedDict):
    """State for document processing workflow."""
    messages: List[Any]
    filename: str
    file_content: Optional[bytes]
    text_content: Optional[str]
    file_type_info: Optional[Dict[str, Any]]
    classification: Optional[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]]
    routing_decision: Optional[Dict[str, Any]]
    notifications_sent: bool
    processing_status: str
    error: Optional[str]


class ProcessingResult(BaseModel):
    """Result of document processing."""
    success: bool
    filename: str
    file_type: Optional[str] = None
    document_type: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    routed_to: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    notifications_sent: bool = False
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


def get_model():
    """Get the LLM model for agents with automatic key rotation."""
    from services.llm_provider import get_model as get_llm_model
    return get_llm_model(model_name="gemini-2.5-flash-lite")


def create_document_processing_workflow():
    """
    Create the multi-agent document processing workflow.
    
    Returns:
        A compiled LangGraph workflow for document processing.
    """
    model = get_model()
    
    # Create specialized agents
    classification_agent = create_react_agent(
        model=model,
        tools=CLASSIFICATION_TOOLS,
        name="classification_expert",
        prompt="""You are a document classification specialist. Your role is to:
1. Detect file types and categories (CAD, document, spreadsheet, etc.)
2. Classify document content into types (invoice, contract, report, etc.)
3. Identify key metadata and priority indicators

Always report your findings in a structured format. Be precise and include confidence scores."""
    )
    
    extraction_agent = create_react_agent(
        model=model,
        tools=EXTRACTION_TOOLS + DOCUMENT_TOOLS,
        name="extraction_expert",
        prompt="""You are a data extraction specialist. Your role is to:
1. Extract structured data from documents based on their type
2. Generate document summaries
3. Process and store documents in the vector database

Use the appropriate extraction tools for different document types.
Always generate a summary for human review."""
    )
    
    routing_agent = create_react_agent(
        model=model,
        tools=ROUTING_TOOLS,
        name="routing_expert",
        prompt="""You are a document routing specialist. Your role is to:
1. Determine which departments should receive a document
2. Assign priority levels
3. Validate routing decisions
4. Send notifications to relevant stakeholders

Consider both file type and content when making routing decisions.
Always send notifications after determining the routing."""
    )
    
    # Create supervisor workflow
    workflow = create_supervisor(
        agents=[classification_agent, extraction_agent, routing_agent],
        model=model,
        prompt="""You are the Document Processing Supervisor managing a team of specialist agents:

1. **classification_expert**: Analyzes file types and classifies document content
2. **extraction_expert**: Extracts structured data and processes documents  
3. **routing_expert**: Routes documents to departments and sends notifications

Your job is to orchestrate these agents to fully process incoming documents:

WORKFLOW:
1. First, have classification_expert analyze the document to determine its type
2. Then, have extraction_expert extract structured data and generate a summary
3. Finally, have routing_expert determine routing and send notifications

For each document, ensure ALL THREE steps are completed in order.
Report the final processing result including:
- Document classification
- Extracted data summary
- Routing decisions and notifications sent

Coordinate efficiently and ensure complete processing of each document."""
    )
    
    return workflow.compile()


async def process_document_with_agents(
    filename: str,
    text_content: str,
    file_content: Optional[bytes] = None
) -> ProcessingResult:
    """
    Process a document using the multi-agent workflow.
    
    Args:
        filename: Name of the file being processed
        text_content: Extracted text content from the document
        file_content: Optional raw file bytes
    
    Returns:
        ProcessingResult with all processing details
    """
    import logging
    logger = logging.getLogger(__name__)
    
    start_time = datetime.now()
    max_workflow_retries = 3
    last_error = None
    
    for retry_attempt in range(max_workflow_retries):
        try:
            # Create a fresh workflow each retry (gets new model with fresh key)
            workflow = create_document_processing_workflow()
            
            # Prepare the input message
            input_message = f"""Process this document completely:

**Filename:** {filename}

**Document Content:**
{text_content[:5000]}  # Limit content length for LLM context

Please:
1. Classify the document type and file category
2. Extract structured data based on the document type
3. Determine routing to appropriate departments
4. Send notifications to the target departments

Report all findings and actions taken."""

            # Run the workflow
            result = workflow.invoke({
                "messages": [
                    HumanMessage(content=input_message)
                ]
            })
            
            # Extract results from messages
            final_message = result["messages"][-1] if result.get("messages") else None
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Parse the result (simplified - in production would parse more carefully)
            return ProcessingResult(
                success=True,
                filename=filename,
                file_type=_extract_file_type(final_message),
                document_type=_extract_document_type(final_message),
                extracted_data=_extract_data(final_message),
                routed_to=_extract_routing(final_message),
                summary=_extract_summary(final_message),
                notifications_sent=True,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check if it's a rate limit error that might be worth retrying
            is_rate_limit = any(term in error_str for term in ["429", "quota", "rate", "resource"])
            
            if is_rate_limit and retry_attempt < max_workflow_retries - 1:
                logger.warning(f"Rate limit hit at workflow level (attempt {retry_attempt + 1}/{max_workflow_retries}), retrying with fresh workflow...")
                import asyncio
                await asyncio.sleep(2 ** retry_attempt)  # Exponential backoff
                continue
            
            # Non-retryable error or max retries reached
            break
    
    # All retries failed
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    return ProcessingResult(
        success=False,
        filename=filename,
        error=str(last_error),
        processing_time_ms=processing_time
    )


def _extract_file_type(message) -> Optional[str]:
    """Extract file type from agent response."""
    if message and hasattr(message, 'content'):
        content = str(message.content).lower()
        for ft in ['pdf', 'docx', 'xlsx', 'pptx', 'dwg', 'csv', 'txt']:
            if ft in content:
                return ft
    return None


def _extract_document_type(message) -> Optional[str]:
    """Extract document type from agent response."""
    if message and hasattr(message, 'content'):
        content = str(message.content).lower()
        for dt in ['invoice', 'contract', 'report', 'specification', 'safety', 'memo']:
            if dt in content:
                return dt
    return None


def _extract_data(message) -> Optional[Dict[str, Any]]:
    """Extract structured data from agent response."""
    # Simplified - would parse JSON from response in production
    return {"extracted": True, "source": "agent_workflow"}


def _extract_routing(message) -> List[str]:
    """Extract routing decisions from agent response."""
    if message and hasattr(message, 'content'):
        content = str(message.content).lower()
        departments = []
        for dept in ['engineering', 'finance', 'legal', 'operations', 'safety', 'compliance', 'hr', 'management', 'procurement']:
            if dept in content:
                departments.append(dept)
        return departments
    return []


def _extract_summary(message) -> Optional[str]:
    """Extract summary from agent response."""
    if message and hasattr(message, 'content'):
        content = str(message.content)
        # Return first 500 chars as summary
        return content[:500] if len(content) > 500 else content
    return None


# Convenience function for single document processing
async def process_single_document(
    filename: str,
    text_content: str
) -> Dict[str, Any]:
    """
    Simple interface to process a single document.
    
    Args:
        filename: Document filename
        text_content: Extracted text content
    
    Returns:
        Dictionary with processing results
    """
    result = await process_document_with_agents(filename, text_content)
    return result.model_dump()
