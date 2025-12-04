"""
Document Processing Schemas - Pydantic models for document operations.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a processed document."""
    title: Optional[str] = None
    source: Optional[str] = None
    page_number: Optional[int] = None
    keywords: List[str] = Field(default_factory=list)
    custom: Dict[str, Any] = Field(default_factory=dict)


class ProcessedNode(BaseModel):
    """A processed document node/chunk."""
    id: str
    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    embedding_dim: Optional[int] = None


class DocumentProcessingRequest(BaseModel):
    """Request for processing a document."""
    collection_name: str = Field(default="documents", description="Target collection name")
    chunk_size: int = Field(default=1024, ge=100, le=8192, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, le=1024, description="Overlap between chunks")
    extract_keywords: bool = Field(default=True, description="Whether to extract keywords")


class DocumentProcessingResponse(BaseModel):
    """Response from document processing."""
    success: bool
    message: str
    document_id: Optional[str] = None
    num_nodes: Optional[int] = None
    nodes: List[ProcessedNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single search result."""
    text: str
    score: Optional[float] = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class SearchResponse(BaseModel):
    """Response from document search."""
    query: str
    answer: Optional[str] = None
    results: List[SearchResult] = Field(default_factory=list)
    total_results: int = 0


class CollectionStats(BaseModel):
    """Statistics for a document collection."""
    name: str
    document_count: int
    total_nodes: int = 0
    embedding_dimension: Optional[int] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
