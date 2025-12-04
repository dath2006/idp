"""
Document Processing Service using LangChain Tools and LlamaIndex.

This service provides document processing capabilities including:
- Document ingestion and chunking
- Embedding generation
- Vector store management
- Document retrieval
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import KeywordExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb

# Configure storage paths
STORAGE_DIR = Path("./storage")
CHROMA_DB_PATH = STORAGE_DIR / "chroma_db"
UPLOADS_DIR = STORAGE_DIR / "uploads"

# Ensure directories exist
STORAGE_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Configure LlamaIndex settings to use HuggingFace embeddings (free, no API key needed)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


class DocumentProcessingResult(BaseModel):
    """Result of document processing operation."""
    success: bool
    message: str
    document_id: Optional[str] = None
    num_nodes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentQueryResult(BaseModel):
    """Result of document query operation."""
    success: bool
    results: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = ""


# Initialize ChromaDB client
def get_chroma_client():
    """Get or create ChromaDB persistent client."""
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def get_vector_store(collection_name: str = "documents"):
    """Get or create ChromaDB vector store."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=collection)


def get_ingestion_pipeline(vector_store: ChromaVectorStore) -> IngestionPipeline:
    """Create an ingestion pipeline with transformations."""
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
            Settings.embed_model,  # Use the configured embedding model
        ],
        vector_store=vector_store,
    )
    return pipeline


# ============================================================
# LangChain Tools for Document Processing
# ============================================================

@tool
def process_document_from_path(
    file_path: str,
    collection_name: str = "documents",
) -> str:
    """
    Process a document from a file path and store it in the vector database.
    
    Args:
        file_path: The path to the document file to process.
        collection_name: The name of the collection to store the document in.
    
    Returns:
        A message indicating the result of the processing operation.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"
        
        # Load document using SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[str(path)])
        documents = reader.load_data()
        
        if not documents:
            return f"Error: No content could be extracted from {file_path}"
        
        # Get vector store and create pipeline
        vector_store = get_vector_store(collection_name)
        pipeline = get_ingestion_pipeline(vector_store)
        
        # Process documents
        nodes = pipeline.run(documents=documents, show_progress=True)
        
        return f"Successfully processed document '{path.name}'. Created {len(nodes)} nodes in collection '{collection_name}'."
    
    except Exception as e:
        return f"Error processing document: {str(e)}"


@tool
def process_text_content(
    content: str,
    document_title: str = "Untitled Document",
    collection_name: str = "documents",
) -> str:
    """
    Process raw text content and store it in the vector database.
    
    Args:
        content: The text content to process.
        document_title: A title for the document.
        collection_name: The name of the collection to store the document in.
    
    Returns:
        A message indicating the result of the processing operation.
    """
    try:
        if not content or not content.strip():
            return "Error: Empty content provided"
        
        # Create a LlamaIndex Document from text
        document = Document(
            text=content,
            metadata={"title": document_title}
        )
        
        # Get vector store and create pipeline
        vector_store = get_vector_store(collection_name)
        pipeline = get_ingestion_pipeline(vector_store)
        
        # Process document
        nodes = pipeline.run(documents=[document], show_progress=True)
        
        return f"Successfully processed text document '{document_title}'. Created {len(nodes)} nodes in collection '{collection_name}'."
    
    except Exception as e:
        return f"Error processing text content: {str(e)}"


@tool
def search_documents(
    query: str,
    collection_name: str = "documents",
    top_k: int = 5,
) -> str:
    """
    Search for relevant documents in the vector database.
    
    Args:
        query: The search query.
        collection_name: The name of the collection to search in.
        top_k: Number of top results to return.
    
    Returns:
        A formatted string containing the search results.
    """
    try:
        vector_store = get_vector_store(collection_name)
        
        # Create index from vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        # Create query engine and search
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)
        
        # Format results
        results = []
        results.append(f"Query: {query}")
        results.append(f"Response: {response.response}")
        results.append("\nSource Documents:")
        
        for i, node in enumerate(response.source_nodes, 1):
            score = getattr(node, 'score', 'N/A')
            text_preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
            results.append(f"\n{i}. Score: {score}")
            results.append(f"   Text: {text_preview}")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"


@tool
def list_collections() -> str:
    """
    List all available document collections in the vector database.
    
    Returns:
        A formatted string containing the list of collections.
    """
    try:
        client = get_chroma_client()
        collections = client.list_collections()
        
        if not collections:
            return "No collections found in the database."
        
        result = ["Available collections:"]
        for collection in collections:
            count = collection.count()
            result.append(f"  - {collection.name}: {count} documents")
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error listing collections: {str(e)}"


@tool
def delete_collection(collection_name: str) -> str:
    """
    Delete a document collection from the vector database.
    
    Args:
        collection_name: The name of the collection to delete.
    
    Returns:
        A message indicating the result of the deletion operation.
    """
    try:
        client = get_chroma_client()
        client.delete_collection(collection_name)
        return f"Successfully deleted collection '{collection_name}'."
    
    except Exception as e:
        return f"Error deleting collection: {str(e)}"


@tool
def get_collection_info(collection_name: str = "documents") -> str:
    """
    Get information about a specific document collection.
    
    Args:
        collection_name: The name of the collection to get info for.
    
    Returns:
        A formatted string containing collection information.
    """
    try:
        client = get_chroma_client()
        collection = client.get_collection(collection_name)
        
        count = collection.count()
        metadata = collection.metadata or {}
        
        result = [
            f"Collection: {collection_name}",
            f"Document count: {count}",
            f"Metadata: {metadata}"
        ]
        
        return "\n".join(result)
    
    except Exception as e:
        return f"Error getting collection info: {str(e)}"


# ============================================================
# Service Functions (for FastAPI endpoints)
# ============================================================

async def process_uploaded_file(
    file_content: bytes,
    filename: str,
    collection_name: str = "documents"
) -> DocumentProcessingResult:
    """
    Process an uploaded file and store it in the vector database.
    
    Args:
        file_content: The raw bytes of the uploaded file.
        filename: The original filename.
        collection_name: The name of the collection to store the document in.
    
    Returns:
        DocumentProcessingResult with processing details.
    """
    try:
        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / filename
        
        try:
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            # Load document
            reader = SimpleDirectoryReader(input_files=[str(temp_path)])
            documents = reader.load_data()
            
            if not documents:
                return DocumentProcessingResult(
                    success=False,
                    message=f"No content could be extracted from {filename}"
                )
            
            # Get vector store and create pipeline
            vector_store = get_vector_store(collection_name)
            pipeline = get_ingestion_pipeline(vector_store)
            
            # Process documents
            nodes = pipeline.run(documents=documents, show_progress=True)
            
            return DocumentProcessingResult(
                success=True,
                message=f"Successfully processed document '{filename}'",
                document_id=filename,
                num_nodes=len(nodes),
                metadata={
                    "collection": collection_name,
                    "original_filename": filename,
                    "num_source_documents": len(documents)
                }
            )
        
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        return DocumentProcessingResult(
            success=False,
            message=f"Error processing document: {str(e)}"
        )


async def process_text(
    content: str,
    title: str = "Untitled",
    collection_name: str = "documents"
) -> DocumentProcessingResult:
    """
    Process text content and store it in the vector database.
    
    Args:
        content: The text content to process.
        title: A title for the document.
        collection_name: The name of the collection to store the document in.
    
    Returns:
        DocumentProcessingResult with processing details.
    """
    try:
        if not content or not content.strip():
            return DocumentProcessingResult(
                success=False,
                message="Empty content provided"
            )
        
        # Create a LlamaIndex Document
        document = Document(
            text=content,
            metadata={"title": title}
        )
        
        # Get vector store and create pipeline
        vector_store = get_vector_store(collection_name)
        pipeline = get_ingestion_pipeline(vector_store)
        
        # Process document
        nodes = pipeline.run(documents=[document], show_progress=True)
        
        return DocumentProcessingResult(
            success=True,
            message=f"Successfully processed text document '{title}'",
            document_id=title,
            num_nodes=len(nodes),
            metadata={
                "collection": collection_name,
                "title": title,
                "content_length": len(content)
            }
        )
    
    except Exception as e:
        return DocumentProcessingResult(
            success=False,
            message=f"Error processing text: {str(e)}"
        )


async def query_documents(
    query: str,
    collection_name: str = "documents",
    top_k: int = 5
) -> DocumentQueryResult:
    """
    Query documents in the vector database.
    
    Args:
        query: The search query.
        collection_name: The name of the collection to search in.
        top_k: Number of top results to return.
    
    Returns:
        DocumentQueryResult with search results.
    """
    try:
        vector_store = get_vector_store(collection_name)
        
        # Create index from vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        # Create query engine and search
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)
        
        # Format results
        results = []
        for node in response.source_nodes:
            results.append({
                "text": node.text,
                "score": getattr(node, 'score', None),
                "metadata": node.metadata if hasattr(node, 'metadata') else {}
            })
        
        return DocumentQueryResult(
            success=True,
            results=results,
            message=str(response.response)
        )
    
    except Exception as e:
        return DocumentQueryResult(
            success=False,
            results=[],
            message=f"Error querying documents: {str(e)}"
        )


# Export all tools for use with LangChain agents
DOCUMENT_TOOLS = [
    process_document_from_path,
    process_text_content,
    search_documents,
    list_collections,
    delete_collection,
    get_collection_info,
]
