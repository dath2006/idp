"""
Agents package for IDP Multi-Agent System.

This package contains specialized agents that work together to process
documents intelligently:

1. DocumentProcessorAgent - Main orchestrator (supervisor)
2. ClassificationAgent - File type and document classification
3. ExtractionAgent - Structured data extraction from documents
4. RoutingAgent - Department routing decisions
"""

from .orchestrator import (
    create_document_processing_workflow,
    process_document_with_agents,
    DocumentProcessingState,
)

__all__ = [
    "create_document_processing_workflow",
    "process_document_with_agents",
    "DocumentProcessingState",
]
