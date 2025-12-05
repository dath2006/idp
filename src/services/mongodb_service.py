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


class DocumentPriority(str, Enum):
    """Priority level of a document."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TeamActionType(str, Enum):
    """Types of actions a team member can take on a document."""
    VIEW = "view"
    APPROVE = "approve"
    REJECT = "reject"
    FORWARD = "forward"
    COMMENT = "comment"
    DOWNLOAD = "download"
    ARCHIVE = "archive"


class TeamStatus(str, Enum):
    """Status of a document for a specific team."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FORWARDED = "forwarded"


class UserRole(str, Enum):
    """User role levels."""
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


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


class TeamStatusRecord(BaseModel):
    """Status of document for a specific team."""
    team: str
    status: TeamStatus = TeamStatus.PENDING
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None


class DocumentAction(BaseModel):
    """Action taken on a document by a team member."""
    id: Optional[str] = Field(default=None, alias="_id")
    document_id: str
    action_type: TeamActionType
    performed_by: str  # user_id
    performed_by_name: str  # user display name
    team: str
    comment: Optional[str] = None
    forwarded_to: Optional[str] = None  # team name if forwarded
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class User(BaseModel):
    """User model for authentication and team membership."""
    id: Optional[str] = Field(default=None, alias="_id")
    email: str
    password_hash: str
    name: str
    teams: List[str] = Field(default_factory=list)  # Team memberships
    role: UserRole = UserRole.MEMBER
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


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
    classification_category: Optional[str] = None  # Comma-separated for multilabel
    classification_categories: List[str] = Field(default_factory=list)  # List of categories
    classification_confidence: Optional[float] = None
    classification_model: str = "multilabel-tfidf"
    classified_at: Optional[datetime] = None
    
    # Team assignment (from classification)
    assigned_teams: List[str] = Field(default_factory=list)  # Teams this doc is assigned to
    team_statuses: List[TeamStatusRecord] = Field(default_factory=list)  # Status per team
    
    # Priority and archival
    priority: DocumentPriority = DocumentPriority.MEDIUM
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    archived_by: Optional[str] = None
    
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
        
        # Index for team-based queries
        await documents.create_index("assigned_teams")
        
        # Index for archived documents
        await documents.create_index("is_archived")
        
        # Compound index for review queue
        await documents.create_index([
            ("status", 1),
            ("created_at", -1)
        ])
        
        # Compound index for team + status queries
        await documents.create_index([
            ("assigned_teams", 1),
            ("status", 1),
            ("is_archived", 1)
        ])
        
        # Users collection indexes
        users = self._db["users"]
        await users.create_index("email", unique=True)
        await users.create_index("teams")
        
        # Document actions collection indexes
        actions = self._db["document_actions"]
        await actions.create_index("document_id")
        await actions.create_index("team")
        await actions.create_index("performed_by")
        await actions.create_index([
            ("document_id", 1),
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
    
    @property
    def users(self) -> AsyncIOMotorCollection:
        """Get the users collection."""
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db["users"]
    
    @property
    def document_actions(self) -> AsyncIOMotorCollection:
        """Get the document actions collection."""
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db["document_actions"]
    
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
    # User Operations
    # ============================================================
    
    async def create_user(self, user: User) -> str:
        """Create a new user."""
        user_dict = user.model_dump(by_alias=True, exclude={"id"})
        user_dict["created_at"] = datetime.utcnow()
        
        result = await self.users.insert_one(user_dict)
        return str(result.inserted_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        doc = await self.users.find_one({"email": email})
        if doc:
            doc["_id"] = str(doc["_id"])
            return User(**doc)
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        doc = await self.users.find_one({"_id": ObjectId(user_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
            return User(**doc)
        return None
    
    async def update_user_login(self, user_id: str) -> bool:
        """Update user's last login time."""
        result = await self.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    async def get_users_by_team(self, team: str) -> List[User]:
        """Get all users in a team."""
        cursor = self.users.find({"teams": team, "is_active": True})
        users = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            users.append(User(**doc))
        return users
    
    async def get_all_users(self) -> List[User]:
        """Get all active users."""
        cursor = self.users.find({"is_active": True})
        users = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            users.append(User(**doc))
        return users
    
    # ============================================================
    # Document Action Operations
    # ============================================================
    
    async def create_action(self, action: DocumentAction) -> str:
        """Record a document action."""
        action_dict = action.model_dump(by_alias=True, exclude={"id"})
        action_dict["created_at"] = datetime.utcnow()
        
        result = await self.document_actions.insert_one(action_dict)
        return str(result.inserted_id)
    
    async def get_document_actions(
        self,
        document_id: str,
        limit: int = 50
    ) -> List[DocumentAction]:
        """Get actions for a document."""
        cursor = self.document_actions.find(
            {"document_id": document_id}
        ).sort("created_at", -1).limit(limit)
        
        actions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            actions.append(DocumentAction(**doc))
        return actions
    
    async def get_team_actions(
        self,
        team: str,
        limit: int = 100
    ) -> List[DocumentAction]:
        """Get recent actions for a team."""
        cursor = self.document_actions.find(
            {"team": team}
        ).sort("created_at", -1).limit(limit)
        
        actions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            actions.append(DocumentAction(**doc))
        return actions
    
    # ============================================================
    # Team-Based Document Operations
    # ============================================================
    
    async def get_documents_by_team(
        self,
        team: str,
        status: Optional[str] = None,
        include_archived: bool = False,
        limit: int = 100,
        skip: int = 0
    ) -> List[DocumentMetadata]:
        """Get documents assigned to a specific team."""
        query = {"assigned_teams": team}
        
        if not include_archived:
            query["is_archived"] = False
        
        if status:
            query["team_statuses"] = {
                "$elemMatch": {"team": team, "status": status}
            }
        
        cursor = self.documents.find(query).sort(
            "created_at", -1
        ).skip(skip).limit(limit)
        
        documents = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            documents.append(DocumentMetadata(**doc))
        return documents
    
    async def count_documents_by_team(
        self,
        team: str,
        include_archived: bool = False
    ) -> int:
        """Count documents assigned to a team."""
        query = {"assigned_teams": team}
        if not include_archived:
            query["is_archived"] = False
        return await self.documents.count_documents(query)
    
    async def update_team_status(
        self,
        document_id: str,
        team: str,
        status: TeamStatus,
        updated_by: str
    ) -> bool:
        """Update the status of a document for a specific team."""
        now = datetime.utcnow()
        
        # First, try to update existing team status
        result = await self.documents.update_one(
            {
                "_id": ObjectId(document_id),
                "team_statuses.team": team
            },
            {
                "$set": {
                    "team_statuses.$.status": status.value,
                    "team_statuses.$.updated_at": now,
                    "team_statuses.$.updated_by": updated_by,
                    "updated_at": now
                }
            }
        )
        
        if result.modified_count == 0:
            # Team status doesn't exist, add it
            new_status = TeamStatusRecord(
                team=team,
                status=status,
                updated_by=updated_by
            )
            result = await self.documents.update_one(
                {"_id": ObjectId(document_id)},
                {
                    "$push": {"team_statuses": new_status.model_dump()},
                    "$set": {"updated_at": now}
                }
            )
        
        return result.modified_count > 0
    
    async def set_assigned_teams(
        self,
        document_id: str,
        teams: List[str]
    ) -> bool:
        """Set the assigned teams for a document."""
        now = datetime.utcnow()
        
        # Create initial team statuses
        team_statuses = [
            TeamStatusRecord(team=team).model_dump()
            for team in teams
        ]
        
        result = await self.documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "assigned_teams": teams,
                    "team_statuses": team_statuses,
                    "classification_categories": teams,
                    "updated_at": now
                }
            }
        )
        return result.modified_count > 0
    
    async def archive_document(
        self,
        document_id: str,
        archived_by: str
    ) -> bool:
        """Archive a document."""
        result = await self.documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "is_archived": True,
                    "archived_at": datetime.utcnow(),
                    "archived_by": archived_by,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
    
    async def get_team_statistics(self, team: str) -> Dict[str, Any]:
        """Get statistics for a specific team."""
        base_query = {"assigned_teams": team, "is_archived": False}
        
        # Total documents
        total = await self.documents.count_documents(base_query)
        
        # By team status
        pipeline = [
            {"$match": base_query},
            {"$unwind": "$team_statuses"},
            {"$match": {"team_statuses.team": team}},
            {"$group": {
                "_id": "$team_statuses.status",
                "count": {"$sum": 1}
            }}
        ]
        
        status_counts = {}
        async for doc in self.documents.aggregate(pipeline):
            status_counts[doc["_id"]] = doc["count"]
        
        # Pending count
        pending = status_counts.get("pending", 0)
        
        return {
            "team": team,
            "total_documents": total,
            "pending": pending,
            "in_review": status_counts.get("in_review", 0),
            "approved": status_counts.get("approved", 0),
            "rejected": status_counts.get("rejected", 0),
            "forwarded": status_counts.get("forwarded", 0),
            "by_status": status_counts
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
