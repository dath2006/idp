"""
Privacy-First Document Schemas.

Pydantic models for the privacy-first document routing API.
Designed for frontend compatibility with proper serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================
# Enums (matching MongoDB service)
# ============================================================

class DocumentSourceEnum(str, Enum):
    """Origin source of the document."""
    EMAIL = "email"
    TELEGRAM = "telegram"
    MANUAL_UPLOAD = "manual_upload"
    API = "api"


class DocumentStatusEnum(str, Enum):
    """Processing status of the document."""
    RECEIVED = "received"
    CLASSIFIED = "classified"
    PENDING_REVIEW = "pending_review"
    ROUTED = "routed"
    DELIVERED = "delivered"
    FAILED = "failed"


class ReviewReasonEnum(str, Enum):
    """Reason document needs human review."""
    NO_DEPARTMENT_MATCH = "no_department_match"
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_MATCHES = "multiple_matches"
    CLASSIFICATION_ERROR = "classification_error"
    MANUAL_REVIEW_REQUESTED = "manual_review_requested"


class DepartmentEnum(str, Enum):
    """Available departments."""
    ENGINEERING = "engineering"
    FINANCE = "finance"
    PROCUREMENT = "procurement"
    HR = "hr"
    OPERATIONS = "operations"
    SAFETY = "safety"
    COMPLIANCE = "compliance"
    LEGAL = "legal"
    MANAGEMENT = "management"


# ============================================================
# Response Models for Frontend
# ============================================================

class FileOriginResponse(BaseModel):
    """File origin information for API responses."""
    source: DocumentSourceEnum
    source_id: Optional[str] = None
    sender_email: Optional[str] = None
    sender_telegram_username: Optional[str] = None
    chat_title: Optional[str] = None
    received_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RoutingInfoResponse(BaseModel):
    """Routing information for API responses."""
    department_id: str
    department_name: str
    department_email: str
    routed_at: datetime
    delivery_status: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HumanReviewResponse(BaseModel):
    """Human review information for API responses."""
    reason: ReviewReasonEnum
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_decision: Optional[str] = None
    review_notes: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentResponse(BaseModel):
    """
    Full document response for API.
    
    Designed for frontend display - includes all metadata
    but NEVER includes document content.
    """
    id: str
    filename: str
    file_extension: str
    file_size_bytes: int
    file_size_display: str  # Human-readable size
    file_path: str
    
    # Origin
    origin: FileOriginResponse
    
    # Classification
    classification_category: Optional[str] = None
    classification_confidence: Optional[float] = None
    classification_model: str = "distilbert-custom"
    
    # Status
    status: DocumentStatusEnum
    
    # Routing
    routed_to: List[RoutingInfoResponse] = Field(default_factory=list)
    
    # Human review
    human_review: Optional[HumanReviewResponse] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    documents: List[DocumentResponse]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool


# ============================================================
# Request Models
# ============================================================

class ManualUploadRequest(BaseModel):
    """Request for manual document upload via API."""
    description: str = Field(
        ..., 
        description="Description of the document (used for classification)"
    )
    sender_name: Optional[str] = Field(
        default=None,
        description="Name of the person submitting"
    )
    sender_email: Optional[str] = Field(
        default=None,
        description="Email for notifications"
    )


class HumanReviewDecision(BaseModel):
    """Human review decision request."""
    document_id: str
    department_id: str = Field(
        ...,
        description="Department to route to, or 'discard'"
    )
    reviewed_by: str = Field(
        ...,
        description="Name/ID of the reviewer"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional review notes"
    )


class BulkReviewDecision(BaseModel):
    """Bulk review decision for multiple documents."""
    decisions: List[HumanReviewDecision]


# ============================================================
# Processing Response Models
# ============================================================

class ProcessingResultResponse(BaseModel):
    """Response from document processing."""
    success: bool
    document_id: str
    filename: str
    
    # Classification (supports multilabel)
    categories: List[str] = Field(default_factory=list)  # All predicted categories
    category: Optional[str] = None  # Primary category (backward compatible)
    confidence: Optional[float] = None
    
    # Routing
    routed_to: List[str] = Field(default_factory=list)
    
    # Human review
    needs_review: bool = False
    review_reason: Optional[str] = None
    
    # Status
    status: str
    error: Optional[str] = None


# ============================================================
# Statistics Models
# ============================================================

class DocumentStatistics(BaseModel):
    """Document processing statistics."""
    total_documents: int
    pending_review: int
    by_status: Dict[str, int]
    by_source: Dict[str, int]
    by_department: Dict[str, int] = Field(default_factory=dict)
    
    # Time-based stats
    today_count: int = 0
    week_count: int = 0
    month_count: int = 0


class ReviewQueueStats(BaseModel):
    """Statistics for the human review queue."""
    pending_count: int
    oldest_pending: Optional[datetime] = None
    by_reason: Dict[str, int] = Field(default_factory=dict)
    average_review_time_hours: Optional[float] = None


# ============================================================
# Health Check Models
# ============================================================

class ServiceStatus(BaseModel):
    """Status of a service component."""
    name: str
    status: str  # healthy, unhealthy, not_configured
    message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class SystemHealthResponse(BaseModel):
    """Overall system health response."""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    services: List[ServiceStatus]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================
# Department Models
# ============================================================

class DepartmentInfo(BaseModel):
    """Department information for frontend."""
    id: str
    name: str
    description: str
    email: str
    priority: int
    document_count: int = 0  # Populated from stats


class DepartmentListResponse(BaseModel):
    """List of all departments."""
    departments: List[DepartmentInfo]


# ============================================================
# Webhook Models
# ============================================================

class TelegramWebhookPayload(BaseModel):
    """Telegram webhook payload (passed through)."""
    update_id: int
    message: Optional[Dict[str, Any]] = None
    edited_message: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from Telegram


class WebhookResponse(BaseModel):
    """Generic webhook response."""
    success: bool
    message: str = ""


# ============================================================
# Utility Functions
# ============================================================

def format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
