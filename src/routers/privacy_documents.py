"""
Privacy-First Document Router.

REST API endpoints for the privacy-first document processing system.
This router handles:
- Document listing and retrieval
- Human review queue management
- Manual document upload
- Statistics and health checks
- Webhook endpoints for email/Telegram

PRIVACY GUARANTEE: No endpoint processes document content.
Classification is based on email body/caption only.
"""

import os
import hashlib
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query
from pydantic import BaseModel

from services.mongodb_service import (
    DocumentMetadata,
    FileOrigin,
    DocumentSource,
    DocumentStatus,
    get_mongodb,
)
from agents.privacy_orchestrator import (
    process_document_privacy_first,
    complete_human_review,
    on_document_received,
)
from schemas.privacy_schema import (
    DocumentResponse,
    DocumentListResponse,
    ProcessingResultResponse,
    ManualUploadRequest,
    HumanReviewDecision,
    BulkReviewDecision,
    DocumentStatistics,
    ReviewQueueStats,
    SystemHealthResponse,
    ServiceStatus,
    DepartmentInfo,
    DepartmentListResponse,
    TelegramWebhookPayload,
    WebhookResponse,
    FileOriginResponse,
    RoutingInfoResponse,
    HumanReviewResponse,
    format_file_size,
    DocumentStatusEnum,
)
from config.departments import DEPARTMENTS, Department


router = APIRouter()


# ============================================================
# Configuration
# ============================================================

MANUAL_UPLOAD_PATH = os.getenv(
    "MANUAL_UPLOAD_PATH",
    "./storage/manual_uploads"
)
Path(MANUAL_UPLOAD_PATH).mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================

def _document_to_response(doc: DocumentMetadata) -> DocumentResponse:
    """Convert MongoDB document to API response."""
    return DocumentResponse(
        id=doc.id or "",
        filename=doc.filename,
        file_extension=doc.file_extension,
        file_size_bytes=doc.file_size_bytes,
        file_size_display=format_file_size(doc.file_size_bytes),
        file_path=doc.file_path,
        origin=FileOriginResponse(
            source=doc.origin.source,
            source_id=doc.origin.source_id,
            sender_email=doc.origin.sender_email,
            sender_telegram_username=doc.origin.sender_telegram_username,
            chat_title=doc.origin.chat_title,
            received_at=doc.origin.received_at,
        ),
        classification_category=doc.classification_category,
        classification_confidence=doc.classification_confidence,
        classification_model=doc.classification_model,
        status=DocumentStatusEnum(doc.status.value),
        routed_to=[
            RoutingInfoResponse(
                department_id=r.department_id,
                department_name=r.department_name,
                department_email=r.department_email,
                routed_at=r.routed_at,
                delivery_status=r.delivery_status,
            )
            for r in doc.routed_to
        ],
        human_review=HumanReviewResponse(
            reason=doc.human_review.reason,
            created_at=doc.human_review.created_at,
            reviewed_at=doc.human_review.reviewed_at,
            reviewed_by=doc.human_review.reviewed_by,
            review_decision=doc.human_review.review_decision,
            review_notes=doc.human_review.review_notes,
        ) if doc.human_review else None,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


# ============================================================
# Document Endpoints
# ============================================================

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
):
    """
    List all documents with pagination.
    
    Supports filtering by status and source.
    """
    mongodb = await get_mongodb()
    
    skip = (page - 1) * per_page
    
    # Get documents based on filters
    if status:
        docs = await mongodb.get_documents_by_status(
            DocumentStatus(status),
            limit=per_page + 1  # +1 to check if there's more
        )
    elif source:
        docs = await mongodb.get_documents_by_source(
            DocumentSource(source),
            limit=per_page + 1
        )
    else:
        docs = await mongodb.get_recent_documents(
            limit=per_page + 1,
            skip=skip
        )
    
    has_next = len(docs) > per_page
    if has_next:
        docs = docs[:per_page]
    
    # Get total count
    stats = await mongodb.get_statistics()
    total = stats.get("total_documents", 0)
    
    return DocumentListResponse(
        documents=[_document_to_response(d) for d in docs],
        total=total,
        page=page,
        per_page=per_page,
        has_next=has_next,
        has_prev=page > 1,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a specific document by ID."""
    mongodb = await get_mongodb()
    doc = await mongodb.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return _document_to_response(doc)


@router.post("/documents/upload", response_model=ProcessingResultResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(...),
    sender_name: Optional[str] = Form(None),
    sender_email: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a document manually via the web interface.
    
    The description will be used for classification (not the document content).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    
    try:
        # Save file to storage
        date_folder = datetime.utcnow().strftime("%Y/%m/%d")
        save_dir = Path(MANUAL_UPLOAD_PATH) / date_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(
            c if c.isalnum() or c in "._-" else "_"
            for c in file.filename
        )
        file_path = save_dir / safe_filename
        
        # Ensure unique
        counter = 1
        while file_path.exists():
            name, ext = os.path.splitext(safe_filename)
            file_path = save_dir / f"{name}_{counter}{ext}"
            counter += 1
        
        # Write file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Calculate hash
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Get extension
        _, ext = os.path.splitext(file.filename)
        
        # Create origin
        origin = FileOrigin(
            source=DocumentSource.MANUAL_UPLOAD,
            sender_email=sender_email,
            original_body=description,  # Used for classification
        )
        
        # Create document metadata
        doc_metadata = DocumentMetadata(
            filename=file.filename,
            file_extension=ext.lower(),
            file_size_bytes=len(content),
            file_path=str(file_path),
            file_hash=file_hash,
            origin=origin,
            status=DocumentStatus.RECEIVED,
        )
        
        # Store in MongoDB
        mongodb = await get_mongodb()
        doc_id = await mongodb.create_document(doc_metadata)
        
        # Process in background
        if background_tasks:
            background_tasks.add_task(on_document_received, doc_id)
        
        return ProcessingResultResponse(
            success=True,
            document_id=doc_id,
            filename=file.filename,
            status="received",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Human Review Endpoints
# ============================================================

@router.get("/review-queue", response_model=DocumentListResponse)
async def get_review_queue(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    """
    Get documents pending human review.
    
    These are documents that couldn't be automatically classified
    with sufficient confidence.
    """
    mongodb = await get_mongodb()
    
    skip = (page - 1) * per_page
    docs = await mongodb.get_review_queue(limit=per_page + 1, skip=skip)
    
    has_next = len(docs) > per_page
    if has_next:
        docs = docs[:per_page]
    
    total = await mongodb.get_review_queue_count()
    
    return DocumentListResponse(
        documents=[_document_to_response(d) for d in docs],
        total=total,
        page=page,
        per_page=per_page,
        has_next=has_next,
        has_prev=page > 1,
    )


@router.post("/review-queue/decide", response_model=ProcessingResultResponse)
async def submit_review_decision(
    decision: HumanReviewDecision,
    background_tasks: BackgroundTasks,
):
    """
    Submit a human review decision for a document.
    
    The reviewer specifies which department to route to,
    or 'discard' to reject the document.
    """
    result = await complete_human_review(
        document_id=decision.document_id,
        reviewed_by=decision.reviewed_by,
        department_id=decision.department_id,
        notes=decision.notes,
    )
    
    return ProcessingResultResponse(
        success=result.success,
        document_id=result.document_id,
        filename=result.filename,
        categories=result.categories or [],
        category=result.category,
        confidence=result.confidence,
        routed_to=result.routed_to,
        status=result.status,
        error=result.error,
    )


@router.post("/review-queue/bulk-decide")
async def submit_bulk_review_decisions(
    bulk: BulkReviewDecision,
    background_tasks: BackgroundTasks,
):
    """Submit multiple review decisions at once."""
    results = []
    
    for decision in bulk.decisions:
        result = await complete_human_review(
            document_id=decision.document_id,
            reviewed_by=decision.reviewed_by,
            department_id=decision.department_id,
            notes=decision.notes,
        )
        results.append({
            "document_id": decision.document_id,
            "success": result.success,
            "status": result.status,
        })
    
    return {
        "processed": len(results),
        "results": results,
    }


@router.get("/review-queue/stats", response_model=ReviewQueueStats)
async def get_review_queue_stats():
    """Get statistics about the human review queue."""
    mongodb = await get_mongodb()
    
    pending_count = await mongodb.get_review_queue_count()
    pending_docs = await mongodb.get_review_queue(limit=1)
    
    oldest = None
    if pending_docs:
        oldest = pending_docs[0].created_at
    
    return ReviewQueueStats(
        pending_count=pending_count,
        oldest_pending=oldest,
        by_reason={},  # TODO: Aggregate by reason
    )


# ============================================================
# Statistics Endpoints
# ============================================================

@router.get("/statistics", response_model=DocumentStatistics)
async def get_statistics():
    """Get document processing statistics."""
    mongodb = await get_mongodb()
    stats = await mongodb.get_statistics()
    
    return DocumentStatistics(
        total_documents=stats.get("total_documents", 0),
        pending_review=stats.get("pending_review", 0),
        by_status=stats.get("by_status", {}),
        by_source=stats.get("by_source", {}),
    )


# ============================================================
# Department Endpoints
# ============================================================

@router.get("/departments", response_model=DepartmentListResponse)
async def list_departments():
    """Get all available departments."""
    departments = []
    
    for dept_id, config in DEPARTMENTS.items():
        departments.append(DepartmentInfo(
            id=dept_id.value,
            name=config.name,
            description=config.description,
            email=config.email,
            priority=config.priority,
        ))
    
    # Sort by priority
    departments.sort(key=lambda d: d.priority)
    
    return DepartmentListResponse(departments=departments)


# ============================================================
# Webhook Endpoints
# ============================================================

@router.post("/webhooks/telegram", response_model=WebhookResponse)
async def telegram_webhook(
    payload: TelegramWebhookPayload,
    background_tasks: BackgroundTasks,
):
    """
    Telegram webhook endpoint.
    
    Receives updates from Telegram bot and processes documents.
    """
    try:
        from services.privacy_telegram_service import process_telegram_webhook
        
        success = await process_telegram_webhook(payload.model_dump())
        
        return WebhookResponse(
            success=success,
            message="Update processed" if success else "Processing failed"
        )
    except Exception as e:
        return WebhookResponse(
            success=False,
            message=str(e)
        )


@router.post("/webhooks/email-notification")
async def email_notification_webhook(
    background_tasks: BackgroundTasks,
):
    """
    Email notification webhook (for services like SendGrid Inbound Parse).
    
    This is a placeholder - implement based on your email provider.
    """
    # TODO: Implement based on email provider webhook format
    return WebhookResponse(
        success=True,
        message="Email webhook placeholder"
    )


# ============================================================
# Health Check Endpoints
# ============================================================

@router.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """
    System health check.
    
    Checks MongoDB, Telegram, Email, and ML model status.
    """
    services = []
    overall_status = "healthy"
    
    # Check MongoDB
    try:
        mongodb = await get_mongodb()
        await mongodb.get_statistics()
        services.append(ServiceStatus(
            name="mongodb",
            status="healthy",
            message="Connected"
        ))
    except Exception as e:
        services.append(ServiceStatus(
            name="mongodb",
            status="unhealthy",
            message=str(e)
        ))
        overall_status = "unhealthy"
    
    # Check Telegram
    try:
        from services.privacy_telegram_service import check_telegram_connection
        tg_status = await check_telegram_connection()
        services.append(ServiceStatus(
            name="telegram",
            status=tg_status.get("status", "unknown"),
            message=tg_status.get("message", ""),
            details=tg_status
        ))
        if tg_status.get("status") == "not_configured":
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        services.append(ServiceStatus(
            name="telegram",
            status="error",
            message=str(e)
        ))
    
    # Check Email
    try:
        from services.email_receiver_service import check_email_connection
        email_status = await check_email_connection()
        services.append(ServiceStatus(
            name="email",
            status=email_status.get("status", "unknown"),
            message=email_status.get("message", ""),
            details=email_status
        ))
        if email_status.get("status") == "not_configured":
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        services.append(ServiceStatus(
            name="email",
            status="error",
            message=str(e)
        ))
    
    # Check ML Model
    try:
        from ml.distilbert_classifier import get_model_info
        model_info = get_model_info()
        services.append(ServiceStatus(
            name="ml_classifier",
            status="healthy" if model_info.get("is_loaded") else "fallback",
            message="Using DistilBERT" if model_info.get("is_loaded") else "Using rule-based fallback",
            details=model_info
        ))
    except Exception as e:
        services.append(ServiceStatus(
            name="ml_classifier",
            status="error",
            message=str(e)
        ))
    
    return SystemHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services
    )


@router.get("/health/quick")
async def quick_health_check():
    """Quick health check (for load balancers)."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
