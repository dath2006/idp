"""
Document Processing Router - FastAPI endpoints for document operations.

Provides REST API endpoints for:
- Uploading and processing documents
- Processing raw text content
- Querying documents
- Managing collections
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from services.document_service import (
    process_uploaded_file,
    process_text,
    query_documents,
    DocumentProcessingResult,
    DocumentQueryResult,
    get_chroma_client,
    get_collection_info as get_collection_info_service,
    DOCUMENT_TOOLS,
)

router = APIRouter()


# ============================================================
# Request/Response Models
# ============================================================

class TextProcessRequest(BaseModel):
    """Request model for processing text content."""
    content: str = Field(..., description="The text content to process")
    title: str = Field(default="Untitled", description="Title for the document")
    collection_name: str = Field(default="documents", description="Collection to store the document")


class QueryRequest(BaseModel):
    """Request model for querying documents."""
    query: str = Field(..., description="The search query")
    collection_name: str = Field(default="documents", description="Collection to search in")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class CollectionInfo(BaseModel):
    """Response model for collection information."""
    name: str
    document_count: int
    metadata: dict = Field(default_factory=dict)


class CollectionListResponse(BaseModel):
    """Response model for listing collections."""
    collections: List[CollectionInfo]


class ToolInfo(BaseModel):
    """Information about available LangChain tools."""
    name: str
    description: str


class ToolsListResponse(BaseModel):
    """Response model for listing available tools."""
    tools: List[ToolInfo]


# ============================================================
# Endpoints
# ============================================================

@router.post("/upload", response_model=DocumentProcessingResult)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="documents"),
):
    """
    Upload and process a document file.
    
    Supports various file formats including:
    - PDF (.pdf)
    - Word documents (.docx)
    - Text files (.txt)
    - Markdown (.md)
    - And more...
    
    The document will be chunked, embedded, and stored in the vector database.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Read file content
    content = await file.read()
    
    if not content:
        raise HTTPException(status_code=400, detail="Empty file provided")
    
    # Process the file
    result = await process_uploaded_file(
        file_content=content,
        filename=file.filename,
        collection_name=collection_name
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)
    
    return result


@router.post("/text", response_model=DocumentProcessingResult)
async def process_text_content(request: TextProcessRequest):
    """
    Process raw text content and store it in the vector database.
    
    Use this endpoint when you have text content directly (not from a file).
    The text will be chunked, embedded, and stored for later retrieval.
    """
    result = await process_text(
        content=request.content,
        title=request.title,
        collection_name=request.collection_name
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)
    
    return result


@router.post("/query", response_model=DocumentQueryResult)
async def query_documents_endpoint(request: QueryRequest):
    """
    Query the document database for relevant information.
    
    Uses semantic search to find the most relevant document chunks
    based on the query. Returns the top-k most similar results.
    """
    result = await query_documents(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)
    
    return result


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """
    List all available document collections.
    
    Returns information about each collection including
    the number of documents stored.
    """
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        collection_infos = []
        for collection in collections:
            collection_infos.append(CollectionInfo(
                name=collection.name,
                document_count=collection.count(),
                metadata=collection.metadata or {}
            ))
        
        return CollectionListResponse(collections=collection_infos)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@router.get("/collections/{collection_name}", response_model=CollectionInfo)
async def get_collection_info(collection_name: str):
    """
    Get detailed information about a specific collection.
    """
    try:
        client = get_chroma_client()
        collection = client.get_collection(collection_name)
        
        return CollectionInfo(
            name=collection_name,
            document_count=collection.count(),
            metadata=collection.metadata or {}
        )
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")


@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a document collection and all its contents.
    
    Warning: This operation is irreversible!
    """
    try:
        client = get_chroma_client()
        client.delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@router.get("/tools", response_model=ToolsListResponse)
async def list_available_tools():
    """
    List all available LangChain tools for document processing.
    
    These tools can be used with LangChain agents for automated
    document processing workflows.
    """
    tools = []
    for tool in DOCUMENT_TOOLS:
        tools.append(ToolInfo(
            name=tool.name,
            description=tool.description or ""
        ))
    
    return ToolsListResponse(tools=tools)


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the document processing service.
    """
    try:
        # Try to connect to ChromaDB
        client = get_chroma_client()
        collections = client.list_collections()
        
        return {
            "status": "healthy",
            "chroma_db": "connected",
            "collections_count": len(collections)
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
