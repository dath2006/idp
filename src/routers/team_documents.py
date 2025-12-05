"""
Team Documents Router.

Handles team-specific document operations:
- View documents assigned to team
- Take actions (approve, reject, forward, comment)
- Filter and search within team documents
- Document preview
"""

import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services.mongodb_service import (
    get_mongodb,
    DocumentMetadata,
    DocumentAction,
    TeamActionType,
    TeamStatus,
    DocumentPriority,
    User,
)
from services.auth_service import (
    user_has_team_access,
    user_can_modify,
)
from routers.users import get_authenticated_user


router = APIRouter()


# ============================================================
# Response Models
# ============================================================

class DocumentActionResponse(BaseModel):
    """Response for a document action."""
    id: str
    document_id: str
    action_type: str
    performed_by: str
    performed_by_name: str
    team: str
    comment: Optional[str] = None
    forwarded_to: Optional[str] = None
    created_at: datetime


class TeamDocumentResponse(BaseModel):
    """Document response for team view."""
    id: str
    filename: str
    file_extension: str
    file_size_bytes: int
    file_path: str
    
    # Classification
    categories: List[str] = []
    confidence: Optional[float] = None
    
    # Team status
    team_status: str = "pending"
    assigned_teams: List[str] = []
    
    # Priority
    priority: str = "medium"
    
    # Origin
    source: str
    sender: Optional[str] = None
    received_at: datetime
    
    # Timestamps
    created_at: datetime
    updated_at: datetime


class TeamDocumentListResponse(BaseModel):
    """Paginated list of team documents."""
    documents: List[TeamDocumentResponse]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool


class TeamStatisticsResponse(BaseModel):
    """Team statistics."""
    team: str
    total_documents: int
    pending: int
    in_review: int
    approved: int
    rejected: int
    forwarded: int


# ============================================================
# Request Models
# ============================================================

class ActionRequest(BaseModel):
    """Request to take an action on a document."""
    action_type: TeamActionType
    comment: Optional[str] = None
    forward_to_team: Optional[str] = None  # Required if action is FORWARD


class BulkActionRequest(BaseModel):
    """Bulk action request."""
    document_ids: List[str]
    action_type: TeamActionType
    comment: Optional[str] = None


# ============================================================
# Helper Functions
# ============================================================

def document_to_team_response(
    doc: DocumentMetadata,
    team: str
) -> TeamDocumentResponse:
    """Convert DocumentMetadata to TeamDocumentResponse."""
    # Get team-specific status
    team_status = "pending"
    for ts in doc.team_statuses:
        if ts.team == team:
            team_status = ts.status.value if hasattr(ts.status, 'value') else ts.status
            break
    
    # Get sender info
    sender = None
    if doc.origin.sender_email:
        sender = doc.origin.sender_email
    elif doc.origin.sender_telegram_username:
        sender = f"@{doc.origin.sender_telegram_username}"
    
    return TeamDocumentResponse(
        id=doc.id,
        filename=doc.filename,
        file_extension=doc.file_extension,
        file_size_bytes=doc.file_size_bytes,
        file_path=doc.file_path,
        categories=doc.classification_categories or [],
        confidence=doc.classification_confidence,
        team_status=team_status,
        assigned_teams=doc.assigned_teams,
        priority=doc.priority.value if hasattr(doc.priority, 'value') else doc.priority,
        source=doc.origin.source.value if hasattr(doc.origin.source, 'value') else doc.origin.source,
        sender=sender,
        received_at=doc.origin.received_at,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


# ============================================================
# Team Document Endpoints
# ============================================================

@router.get("/teams/{team_id}/documents", response_model=TeamDocumentListResponse)
async def get_team_documents(
    team_id: str,
    status: Optional[str] = Query(None, description="Filter by status: pending, in_review, approved, rejected, forwarded"),
    priority: Optional[str] = Query(None, description="Filter by priority: high, medium, low"),
    include_archived: bool = Query(False, description="Include archived documents"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_authenticated_user)
):
    """
    Get documents assigned to a team.
    
    Only accessible by team members or admins.
    """
    team_id = team_id.lower()
    
    if not user_has_team_access(user, team_id):
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    mongodb = await get_mongodb()
    
    # Get documents
    skip = (page - 1) * per_page
    documents = await mongodb.get_documents_by_team(
        team=team_id,
        status=status,
        include_archived=include_archived,
        limit=per_page,
        skip=skip
    )
    
    # Get total count
    total = await mongodb.count_documents_by_team(team_id, include_archived)
    
    # Convert to response
    doc_responses = [
        document_to_team_response(doc, team_id)
        for doc in documents
    ]
    
    # Filter by priority if specified
    if priority:
        doc_responses = [d for d in doc_responses if d.priority == priority]
    
    return TeamDocumentListResponse(
        documents=doc_responses,
        total=total,
        page=page,
        per_page=per_page,
        has_next=(page * per_page) < total,
        has_prev=page > 1
    )


@router.get("/teams/{team_id}/documents/{document_id}", response_model=TeamDocumentResponse)
async def get_team_document(
    team_id: str,
    document_id: str,
    user: User = Depends(get_authenticated_user)
):
    """
    Get a specific document for a team.
    """
    team_id = team_id.lower()
    
    if not user_has_team_access(user, team_id):
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    mongodb = await get_mongodb()
    doc = await mongodb.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if team_id not in doc.assigned_teams:
        raise HTTPException(status_code=403, detail="Document not assigned to this team")
    
    # Record view action
    action = DocumentAction(
        document_id=document_id,
        action_type=TeamActionType.VIEW,
        performed_by=user.id,
        performed_by_name=user.name,
        team=team_id
    )
    await mongodb.create_action(action)
    
    return document_to_team_response(doc, team_id)


@router.get("/teams/{team_id}/statistics", response_model=TeamStatisticsResponse)
async def get_team_statistics(
    team_id: str,
    user: User = Depends(get_authenticated_user)
):
    """
    Get statistics for a team.
    """
    team_id = team_id.lower()
    
    if not user_has_team_access(user, team_id):
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    mongodb = await get_mongodb()
    stats = await mongodb.get_team_statistics(team_id)
    
    return TeamStatisticsResponse(**stats)


# ============================================================
# Document Action Endpoints
# ============================================================

@router.post("/teams/{team_id}/documents/{document_id}/action", response_model=DocumentActionResponse)
async def take_document_action(
    team_id: str,
    document_id: str,
    action_request: ActionRequest,
    user: User = Depends(get_authenticated_user)
):
    """
    Take an action on a document.
    
    Actions: view, approve, reject, forward, comment, download, archive
    """
    team_id = team_id.lower()
    
    if not user_can_modify(user, team_id):
        raise HTTPException(status_code=403, detail="No permission to modify documents in this team")
    
    mongodb = await get_mongodb()
    doc = await mongodb.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if team_id not in doc.assigned_teams:
        raise HTTPException(status_code=403, detail="Document not assigned to this team")
    
    # Handle action
    action_type = action_request.action_type
    
    # Update team status based on action
    if action_type == TeamActionType.APPROVE:
        await mongodb.update_team_status(document_id, team_id, TeamStatus.APPROVED, user.id)
    elif action_type == TeamActionType.REJECT:
        await mongodb.update_team_status(document_id, team_id, TeamStatus.REJECTED, user.id)
    elif action_type == TeamActionType.FORWARD:
        if not action_request.forward_to_team:
            raise HTTPException(status_code=400, detail="forward_to_team required for forward action")
        
        forward_team = action_request.forward_to_team.lower()
        
        # Add new team to document
        if forward_team not in doc.assigned_teams:
            new_teams = doc.assigned_teams + [forward_team]
            await mongodb.set_assigned_teams(document_id, new_teams)
        
        await mongodb.update_team_status(document_id, team_id, TeamStatus.FORWARDED, user.id)
    elif action_type == TeamActionType.ARCHIVE:
        await mongodb.archive_document(document_id, user.id)
    
    # Record the action
    action = DocumentAction(
        document_id=document_id,
        action_type=action_type,
        performed_by=user.id,
        performed_by_name=user.name,
        team=team_id,
        comment=action_request.comment,
        forwarded_to=action_request.forward_to_team
    )
    action_id = await mongodb.create_action(action)
    
    # Send WebSocket notification
    await _notify_action_performed(
        document_id=document_id,
        team=team_id,
        action_type=action_type.value,
        performed_by=user.name,
        new_status=action_type.value if action_type in [TeamActionType.APPROVE, TeamActionType.REJECT] else None,
        forward_to_team=action_request.forward_to_team
    )
    
    return DocumentActionResponse(
        id=action_id,
        document_id=document_id,
        action_type=action_type.value,
        performed_by=user.id,
        performed_by_name=user.name,
        team=team_id,
        comment=action_request.comment,
        forwarded_to=action_request.forward_to_team,
        created_at=datetime.utcnow()
    )


async def _notify_action_performed(
    document_id: str,
    team: str,
    action_type: str,
    performed_by: str,
    new_status: Optional[str] = None,
    forward_to_team: Optional[str] = None
) -> None:
    """Send WebSocket notification when an action is performed."""
    try:
        from services.websocket_service import (
            manager,
            create_action_performed_notification,
            create_status_changed_notification
        )
        
        # Notify the current team
        notification = create_action_performed_notification(
            document_id=document_id,
            team=team,
            action_type=action_type,
            performed_by=performed_by
        )
        await manager.broadcast_to_team(team, notification)
        
        # If status changed, send status notification
        if new_status:
            status_notification = create_status_changed_notification(
                document_id=document_id,
                team=team,
                new_status=new_status,
                performed_by=performed_by
            )
            await manager.broadcast_to_team(team, status_notification)
        
        # If forwarded to another team, notify that team too
        if forward_to_team:
            forward_notification = create_action_performed_notification(
                document_id=document_id,
                team=forward_to_team,
                action_type="forwarded_to_you",
                performed_by=performed_by
            )
            await manager.broadcast_to_team(forward_to_team, forward_notification)
            
    except Exception as e:
        print(f"⚠️ WebSocket notification failed: {e}")


@router.post("/teams/{team_id}/documents/bulk-action")
async def bulk_action(
    team_id: str,
    bulk_request: BulkActionRequest,
    user: User = Depends(get_authenticated_user)
):
    """
    Take action on multiple documents.
    """
    team_id = team_id.lower()
    
    if not user_can_modify(user, team_id):
        raise HTTPException(status_code=403, detail="No permission to modify documents in this team")
    
    mongodb = await get_mongodb()
    results = []
    
    for doc_id in bulk_request.document_ids:
        try:
            doc = await mongodb.get_document(doc_id)
            if not doc or team_id not in doc.assigned_teams:
                results.append({"document_id": doc_id, "success": False, "error": "Not found or not assigned"})
                continue
            
            # Update status
            if bulk_request.action_type == TeamActionType.APPROVE:
                await mongodb.update_team_status(doc_id, team_id, TeamStatus.APPROVED, user.id)
            elif bulk_request.action_type == TeamActionType.REJECT:
                await mongodb.update_team_status(doc_id, team_id, TeamStatus.REJECTED, user.id)
            
            # Record action
            action = DocumentAction(
                document_id=doc_id,
                action_type=bulk_request.action_type,
                performed_by=user.id,
                performed_by_name=user.name,
                team=team_id,
                comment=bulk_request.comment
            )
            await mongodb.create_action(action)
            
            results.append({"document_id": doc_id, "success": True})
        except Exception as e:
            results.append({"document_id": doc_id, "success": False, "error": str(e)})
    
    return {
        "processed": len(results),
        "results": results
    }


@router.get("/teams/{team_id}/documents/{document_id}/actions", response_model=List[DocumentActionResponse])
async def get_document_actions(
    team_id: str,
    document_id: str,
    limit: int = Query(50, le=100),
    user: User = Depends(get_authenticated_user)
):
    """
    Get action history for a document.
    """
    team_id = team_id.lower()
    
    if not user_has_team_access(user, team_id):
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    mongodb = await get_mongodb()
    actions = await mongodb.get_document_actions(document_id, limit=limit)
    
    return [
        DocumentActionResponse(
            id=a.id,
            document_id=a.document_id,
            action_type=a.action_type.value if hasattr(a.action_type, 'value') else a.action_type,
            performed_by=a.performed_by,
            performed_by_name=a.performed_by_name,
            team=a.team,
            comment=a.comment,
            forwarded_to=a.forwarded_to,
            created_at=a.created_at
        )
        for a in actions
    ]


# ============================================================
# Document Preview Endpoint
# ============================================================

@router.get("/teams/{team_id}/documents/{document_id}/preview")
async def preview_document(
    team_id: str,
    document_id: str,
    user: User = Depends(get_authenticated_user)
):
    """
    Get document file for preview.
    
    Returns the actual file for preview in browser.
    Supported types: PDF, images, text files.
    """
    team_id = team_id.lower()
    
    if not user_has_team_access(user, team_id):
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    mongodb = await get_mongodb()
    doc = await mongodb.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if team_id not in doc.assigned_teams:
        raise HTTPException(status_code=403, detail="Document not assigned to this team")
    
    # Check if file exists
    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine content type
    content_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "application/xml",
        ".html": "text/html",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    
    ext = doc.file_extension.lower()
    content_type = content_types.get(ext, "application/octet-stream")
    
    # Record download action
    action = DocumentAction(
        document_id=document_id,
        action_type=TeamActionType.DOWNLOAD,
        performed_by=user.id,
        performed_by_name=user.name,
        team=team_id
    )
    await mongodb.create_action(action)
    
    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=doc.filename
    )


@router.get("/teams/{team_id}/documents/{document_id}/download")
async def download_document(
    team_id: str,
    document_id: str,
    user: User = Depends(get_authenticated_user)
):
    """
    Download document file.
    
    Returns file as attachment for download.
    """
    team_id = team_id.lower()
    
    if not user_has_team_access(user, team_id):
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    mongodb = await get_mongodb()
    doc = await mongodb.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if team_id not in doc.assigned_teams:
        raise HTTPException(status_code=403, detail="Document not assigned to this team")
    
    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Record download action
    action = DocumentAction(
        document_id=document_id,
        action_type=TeamActionType.DOWNLOAD,
        performed_by=user.id,
        performed_by_name=user.name,
        team=team_id
    )
    await mongodb.create_action(action)
    
    return FileResponse(
        path=str(file_path),
        filename=doc.filename,
        media_type="application/octet-stream"
    )
