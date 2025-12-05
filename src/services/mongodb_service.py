"""
MongoDB Service for Privacy-First Document Metadata Storage.

This service handles all MongoDB operations for storing document metadata
WITHOUT storing actual document content - ensuring compliance and privacy.

Key Features:
- Async MongoDB operations with Motor
- Document metadata tracking with file origin
- Human-in-the-loop queue for unroutable documents
- Full audit trail of document routing
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pydantic import BaseModel, Field
from bson import ObjectId


# ============================================================
# Configuration
# ============================================================

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "idp_privacy")


# ============================================================
# Enums
# ============================================================

class DocumentSource(str, Enum):
    """Origin source of the document."""
    EMAIL = "email"
    TELEGRAM = "telegram"
    MANUAL_UPLOAD = "manual_upload"
    API = "api"


class DocumentStatus(str, Enum):
    """Processing status of the document."""
    RECEIVED = "received"
    CLASSIFIED = "classified"
    PENDING_REVIEW = "pending_review"  # Human-in-the-loop
    ROUTED = "routed"
    DELIVERED = "delivered"
    FAILED = "failed"


class HumanReviewReason(str, Enum):
    """Reason document needs human review."""
    NO_DEPARTMENT_MATCH = "no_department_match"
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_MATCHES = "multiple_matches"
    CLASSIFICATION_ERROR = "classification_error"
    MANUAL_REVIEW_REQUESTED = "manual_review_requested"


# ============================================================
# MongoDB Document Models (Pydantic)
# ============================================================

class FileOrigin(BaseModel):
    """Tracks where a file originally came from."""
    source: DocumentSource
    source_id: Optional[str] = None  # Email message ID, Telegram message ID, etc.
    sender_email: Optional[str] = None
    sender_telegram_id: Optional[str] = None
    sender_telegram_username: Optional[str] = None
    chat_id: Optional[str] = None  # Telegram chat/group ID
    chat_title: Optional[str] = None  # Telegram group name
    received_at: datetime = Field(default_factory=datetime.utcnow)
    original_subject: Optional[str] = None  # Email subject
    original_body: Optional[str] = None  # Email body or Telegram caption (used for classification)


class RoutingInfo(BaseModel):
    """Information about where document was routed."""
    department_id: str
    department_name: str
    department_email: str
    routed_at: datetime = Field(default_factory=datetime.utcnow)
    delivery_status: str = "pending"  # pending, sent, delivered, failed
    delivery_error: Optional[str] = None


class HumanReviewRequest(BaseModel):
    """Request for human review of unroutable document."""
    reason: HumanReviewReason
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_decision: Optional[str] = None  # department_id or "discard"
    review_notes: Optional[str] = None


class DocumentMetadata(BaseModel):
    """
    Complete metadata record for a document.
    
    NOTE: Does NOT contain document content - only metadata for privacy compliance.
    """
    # MongoDB ID (set after insert)
    id: Optional[str] = Field(default=None, alias="_id")
    
    # File information (NO content)
    filename: str
    file_extension: str
    file_size_bytes: int
    file_path: str  # Local path where file is stored
    file_hash: Optional[str] = None  # SHA256 hash for deduplication
    
    # Origin tracking
    origin: FileOrigin
    
    # Classification (from DistilBERT, not LLM)
    classification_category: Optional[str] = None
    classification_confidence: Optional[float] = None
    classification_model: str = "distilbert-custom"
    classified_at: Optional[datetime] = None
    
    # Routing
    status: DocumentStatus = DocumentStatus.RECEIVED
    routed_to: List[RoutingInfo] = Field(default_factory=list)
    
    # Human-in-the-loop
    human_review: Optional[HumanReviewRequest] = None
    
    # Audit trail
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


# ============================================================
# MongoDB Client Singleton
# ============================================================

class MongoDBService:
    """Async MongoDB service for document metadata operations."""
    
    _instance: Optional["MongoDBService"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def connect(self) -> None:
        """Initialize MongoDB connection."""
        if self._client is None:
            self._client = AsyncIOMotorClient(MONGODB_URI)
            self._db = self._client[MONGODB_DATABASE]
            
            # Create indexes
            await self._create_indexes()
            
            print(f"✅ Connected to MongoDB: {MONGODB_DATABASE}")
    
    async def _create_indexes(self) -> None:
        """Create necessary indexes for efficient queries."""
        documents = self._db["documents"]
        
        # Index for status-based queries (human review queue)
        await documents.create_index("status")
        
        # Index for origin source queries
        await documents.create_index("origin.source")
        
        # Index for date-based queries
        await documents.create_index("created_at")
        
        # Index for classification
        await documents.create_index("classification_category")
        
        # Compound index for review queue
        await documents.create_index([
            ("status", 1),
            ("created_at", -1)
        ])
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("🔌 Disconnected from MongoDB")
    
    @property
    def documents(self) -> AsyncIOMotorCollection:
        """Get the documents collection."""
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db["documents"]
    
    @property
    def review_queue(self) -> AsyncIOMotorCollection:
        """Get the human review queue collection."""
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db["review_queue"]
    
    # ============================================================
    # Document CRUD Operations
    # ============================================================
    
    async def create_document(self, metadata: DocumentMetadata) -> str:
        """
        Create a new document metadata record.
        
        Returns the inserted document ID.
        """
        doc_dict = metadata.model_dump(by_alias=True, exclude={"id"})
        doc_dict["created_at"] = datetime.utcnow()
        doc_dict["updated_at"] = datetime.utcnow()
        
        result = await self.documents.insert_one(doc_dict)
        return str(result.inserted_id)
    
    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get a document by ID."""
        doc = await self.documents.find_one({"_id": ObjectId(document_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
            return DocumentMetadata(**doc)
        return None
    
    async def update_document(
        self, 
        document_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update a document's metadata."""
        updates["updated_at"] = datetime.utcnow()
        result = await self.documents.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def update_classification(
        self,
        document_id: str,
        category: str,
        confidence: float,
        model_name: str = "distilbert-custom"
    ) -> bool:
        """Update document classification results."""
        return await self.update_document(document_id, {
            "classification_category": category,
            "classification_confidence": confidence,
            "classification_model": model_name,
            "classified_at": datetime.utcnow(),
            "status": DocumentStatus.CLASSIFIED.value
        })
    
    async def update_routing(
        self,
        document_id: str,
        routing_info: RoutingInfo
    ) -> bool:
        """Add routing information to a document."""
        result = await self.documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$push": {"routed_to": routing_info.model_dump()},
                "$set": {
                    "status": DocumentStatus.ROUTED.value,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    # ============================================================
    # Human-in-the-Loop Operations
    # ============================================================
    
    async def request_human_review(
        self,
        document_id: str,
        reason: HumanReviewReason
    ) -> bool:
        """
        Move document to human review queue.
        
        Called when:
        - No department match found
        - Classification confidence too low
        - Multiple conflicting matches
        """
        review_request = HumanReviewRequest(reason=reason)
        
        result = await self.documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": DocumentStatus.PENDING_REVIEW.value,
                    "human_review": review_request.model_dump(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    async def get_review_queue(
        self,
        limit: int = 50,
        skip: int = 0
    ) -> List[DocumentMetadata]:
        """Get documents pending human review."""
        cursor = self.documents.find(
            {"status": DocumentStatus.PENDING_REVIEW.value}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(DocumentMetadata(**doc))
        
        return documents
    
    async def get_review_queue_count(self) -> int:
        """Get count of documents pending review."""
        return await self.documents.count_documents(
            {"status": DocumentStatus.PENDING_REVIEW.value}
        )
    
    async def complete_review(
        self,
        document_id: str,
        reviewed_by: str,
        decision: str,  # department_id or "discard"
        notes: Optional[str] = None
    ) -> bool:
        """Complete human review of a document."""
        result = await self.documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "human_review.reviewed_at": datetime.utcnow(),
                    "human_review.reviewed_by": reviewed_by,
                    "human_review.review_decision": decision,
                    "human_review.review_notes": notes,
                    "status": DocumentStatus.CLASSIFIED.value if decision != "discard" else DocumentStatus.FAILED.value,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    # ============================================================
    # Query Operations
    # ============================================================
    
    async def get_documents_by_status(
        self,
        status: DocumentStatus,
        limit: int = 100
    ) -> List[DocumentMetadata]:
        """Get documents by status."""
        cursor = self.documents.find(
            {"status": status.value}
        ).sort("created_at", -1).limit(limit)
        
        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(DocumentMetadata(**doc))
        
        return documents
    
    async def get_documents_by_source(
        self,
        source: DocumentSource,
        limit: int = 100
    ) -> List[DocumentMetadata]:
        """Get documents by origin source."""
        cursor = self.documents.find(
            {"origin.source": source.value}
        ).sort("created_at", -1).limit(limit)
        
        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(DocumentMetadata(**doc))
        
        return documents
    
    async def get_recent_documents(
        self,
        limit: int = 50,
        skip: int = 0
    ) -> List[DocumentMetadata]:
        """Get most recent documents."""
        cursor = self.documents.find().sort(
            "created_at", -1
        ).skip(skip).limit(limit)
        
        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(DocumentMetadata(**doc))
        
        return documents
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        pipeline = [
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        status_counts = {}
        async for doc in self.documents.aggregate(pipeline):
            status_counts[doc["_id"]] = doc["count"]
        
        # Get source breakdown
        source_pipeline = [
            {
                "$group": {
                    "_id": "$origin.source",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        source_counts = {}
        async for doc in self.documents.aggregate(source_pipeline):
            source_counts[doc["_id"]] = doc["count"]
        
        total = await self.documents.count_documents({})
        pending_review = await self.get_review_queue_count()
        
        return {
            "total_documents": total,
            "pending_review": pending_review,
            "by_status": status_counts,
            "by_source": source_counts
        }


# ============================================================
# Global Instance
# ============================================================

mongodb_service = MongoDBService()


async def get_mongodb() -> MongoDBService:
    """Get MongoDB service instance (ensures connection)."""
    if mongodb_service._client is None:
        await mongodb_service.connect()
    return mongodb_service
