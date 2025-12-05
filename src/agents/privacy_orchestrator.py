"""
Privacy-First Document Orchestrator.

This orchestrator handles the privacy-safe document routing workflow:
1. Receives document metadata (NOT content)
2. Classifies using DistilBERT based on email body/caption ONLY
3. Routes to appropriate department via email
4. Triggers human-in-the-loop when classification is uncertain

Key Principles:
- NO LLM access to document content
- Classification based on message/email body only
- Human review for uncertain classifications
- Full audit trail in MongoDB
- Supports multilabel classification (documents can belong to multiple departments)
"""

import os
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from services.mongodb_service import (
    DocumentMetadata,
    DocumentStatus,
    RoutingInfo,
    HumanReviewReason,
    get_mongodb,
)
from ml.distilbert_classifier import (
    classify_document,
    ClassificationResult,
    DocumentCategory,
    CONFIDENCE_THRESHOLD,
)
from config.departments import (
    Department,
    DEPARTMENTS,
    get_department_by_id,
)


# ============================================================
# Configuration
# ============================================================

# Confidence threshold below which human review is required
REVIEW_THRESHOLD = float(os.getenv("CLASSIFICATION_REVIEW_THRESHOLD", "0.5"))

# Email for forwarding documents
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")


# ============================================================
# Processing Result
# ============================================================

@dataclass
class PrivacyProcessingResult:
    """Result from privacy-first document processing."""
    success: bool
    document_id: str
    filename: str
    
    # Classification (supports multilabel)
    categories: List[str] = None  # Multiple categories for multilabel
    category: Optional[str] = None  # Primary category (first one)
    confidence: Optional[float] = None
    
    # Routing
    routed_to: List[str] = None
    department_emails: List[str] = None
    
    # Human review
    needs_review: bool = False
    review_reason: Optional[str] = None
    
    # Status
    status: str = "processed"
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.routed_to is None:
            self.routed_to = []
        if self.department_emails is None:
            self.department_emails = []
        # Set primary category from categories list
        if not self.category and self.categories:
            self.category = self.categories[0]


# ============================================================
# Email Forwarding Service
# ============================================================

class EmailForwardingService:
    """
    Service for forwarding documents to departments via email.
    
    Uses SMTP to send emails with document references (not content).
    """
    
    def __init__(self):
        self.server = SMTP_SERVER
        self.port = SMTP_PORT
        self.email = SMTP_EMAIL
        self.password = SMTP_PASSWORD
        self._is_configured = bool(self.email and self.password)
    
    async def forward_document(
        self,
        to_email: str,
        document_metadata: DocumentMetadata,
        classification: ClassificationResult
    ) -> bool:
        """
        Forward document reference to department email.
        
        NOTE: Sends document path/reference, not actual content.
        """
        if not self._is_configured:
            print(f"⚠️ SMTP not configured. Would forward to: {to_email}")
            return True  # Simulate success for development
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.email
            msg["To"] = to_email
            msg["Subject"] = f"[IDP] Document Routed: {document_metadata.filename}"
            
            # Build email body (NO document content)
            categories_str = ", ".join(classification.category_names) if classification.categories else "Unknown"
            body = f"""
A new document has been routed to your department.

Document Details:
- Filename: {document_metadata.filename}
- File Type: {document_metadata.file_extension}
- Size: {document_metadata.file_size_bytes / 1024:.1f} KB
- Received: {document_metadata.origin.received_at.strftime('%Y-%m-%d %H:%M UTC')}

Classification:
- Categories: {categories_str}
- Confidence: {classification.confidence:.0%}
- Model: {classification.model_used}

Source:
- Origin: {document_metadata.origin.source.value}
- From: {document_metadata.origin.sender_email or document_metadata.origin.sender_telegram_username or 'Unknown'}

Document Location:
{document_metadata.file_path}

---
This is an automated message from the IDP Privacy-First System.
Document content was NOT processed - classification based on message text only.
            """.strip()
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send email
            with smtplib.SMTP(self.server, self.port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
            
            print(f"📧 Forwarded to {to_email}: {document_metadata.filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to forward email: {e}")
            return False


# ============================================================
# Privacy Orchestrator
# ============================================================

class PrivacyOrchestrator:
    """
    Main orchestrator for privacy-first document processing.
    
    Workflow:
    1. Load document metadata from MongoDB
    2. Classify using DistilBERT (message/caption only)
    3. If confident → route to department
    4. If uncertain → queue for human review
    5. Send notifications
    """
    
    def __init__(self):
        self.email_forwarder = EmailForwardingService()
    
    async def process_document(self, document_id: str) -> PrivacyProcessingResult:
        """
        Process a document through the privacy-safe pipeline.
        
        Args:
            document_id: MongoDB document ID
            
        Returns:
            PrivacyProcessingResult with processing outcome
        """
        mongodb = await get_mongodb()
        
        # Load document metadata
        doc = await mongodb.get_document(document_id)
        if not doc:
            return PrivacyProcessingResult(
                success=False,
                document_id=document_id,
                filename="unknown",
                error="Document not found"
            )
        
        try:
            # Step 1: Classify based on message/caption ONLY (multilabel)
            classification = classify_document(
                subject=doc.origin.original_subject or "",
                body=doc.origin.original_body or "",
                filename=doc.filename
            )
            
            # Update classification in MongoDB (store all categories)
            categories_str = ",".join(classification.category_names)
            await mongodb.update_classification(
                document_id=document_id,
                category=categories_str,  # Store all categories comma-separated
                confidence=classification.confidence,
                model_name=classification.model_used
            )
            
            # Step 2: Check if human review needed
            if classification.requires_review or classification.confidence < REVIEW_THRESHOLD:
                return await self._request_human_review(
                    doc, 
                    document_id, 
                    classification
                )
            
            # Step 3: Route to all matching departments (multilabel)
            return await self._route_to_departments(
                doc,
                document_id,
                classification
            )
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            await mongodb.update_document(document_id, {
                "status": DocumentStatus.FAILED.value,
                "error": str(e)
            })
            
            return PrivacyProcessingResult(
                success=False,
                document_id=document_id,
                filename=doc.filename,
                error=str(e)
            )
    
    async def _request_human_review(
        self,
        doc: DocumentMetadata,
        document_id: str,
        classification: ClassificationResult
    ) -> PrivacyProcessingResult:
        """Queue document for human review."""
        mongodb = await get_mongodb()
        
        # Determine review reason
        if DocumentCategory.UNKNOWN in classification.categories:
            reason = HumanReviewReason.NO_DEPARTMENT_MATCH
        elif classification.confidence < REVIEW_THRESHOLD:
            reason = HumanReviewReason.LOW_CONFIDENCE
        else:
            reason = HumanReviewReason.CLASSIFICATION_ERROR
        
        # Request human review
        await mongodb.request_human_review(document_id, reason)
        
        print(f"👤 Human review requested: {doc.filename} ({reason.value})")
        
        return PrivacyProcessingResult(
            success=True,
            document_id=document_id,
            filename=doc.filename,
            categories=classification.category_names,
            category=classification.category.value if classification.category else None,
            confidence=classification.confidence,
            needs_review=True,
            review_reason=reason.value,
            status="pending_review"
        )
    
    async def _route_to_departments(
        self,
        doc: DocumentMetadata,
        document_id: str,
        classification: ClassificationResult
    ) -> PrivacyProcessingResult:
        """Route document to all matching departments (multilabel support)."""
        mongodb = await get_mongodb()
        
        routed_to = []
        department_emails = []
        
        # Route to ALL matched categories (multilabel)
        for category in classification.categories:
            if category == DocumentCategory.UNKNOWN:
                continue
                
            dept_id = self._category_to_department(category)
            if not dept_id:
                continue
            
            dept_config = get_department_by_id(dept_id)
            if not dept_config:
                continue
            
            # Forward document reference via email
            email_sent = await self.email_forwarder.forward_document(
                to_email=dept_config.email,
                document_metadata=doc,
                classification=classification
            )
            
            # Record routing in MongoDB
            routing_info = RoutingInfo(
                department_id=dept_id.value,
                department_name=dept_config.name,
                department_email=dept_config.email,
                delivery_status="sent" if email_sent else "failed"
            )
            
            await mongodb.update_routing(document_id, routing_info)
            
            routed_to.append(dept_config.name)
            department_emails.append(dept_config.email)
            
            print(f"📨 Routed to {dept_config.name}: {doc.filename}")
        
        # If no departments were routed to, request human review
        if not routed_to:
            return await self._request_human_review(doc, document_id, classification)
        
        return PrivacyProcessingResult(
            success=True,
            document_id=document_id,
            filename=doc.filename,
            categories=classification.category_names,
            category=classification.category.value if classification.category else None,
            confidence=classification.confidence,
            routed_to=routed_to,
            department_emails=department_emails,
            status="routed"
        )
    
    def _category_to_department(
        self, 
        category: DocumentCategory
    ) -> Optional[Department]:
        """Map classification category to department."""
        # Model outputs: ['Compliance', 'Engineer', 'Finance', 'HR', 'Legal', 'Operations', 'Procurement', 'Project Leader', 'Safety']
        mapping = {
            DocumentCategory.COMPLIANCE: Department.COMPLIANCE,
            DocumentCategory.ENGINEER: Department.ENGINEERING,  # Model says "Engineer" -> Engineering dept
            DocumentCategory.FINANCE: Department.FINANCE,
            DocumentCategory.HR: Department.HR,
            DocumentCategory.LEGAL: Department.LEGAL,
            DocumentCategory.OPERATIONS: Department.OPERATIONS,
            DocumentCategory.PROCUREMENT: Department.PROCUREMENT,
            DocumentCategory.PROJECT_LEADER: Department.MANAGEMENT,  # Project Leader -> Management dept
            DocumentCategory.SAFETY: Department.SAFETY,
        }
        return mapping.get(category)
    
    async def process_human_review_decision(
        self,
        document_id: str,
        reviewed_by: str,
        department_id: str,
        notes: Optional[str] = None
    ) -> PrivacyProcessingResult:
        """
        Process a human review decision.
        
        Called when a human reviewer assigns a department.
        """
        mongodb = await get_mongodb()
        
        # Complete the review
        await mongodb.complete_review(
            document_id=document_id,
            reviewed_by=reviewed_by,
            decision=department_id,
            notes=notes
        )
        
        # Get document
        doc = await mongodb.get_document(document_id)
        if not doc:
            return PrivacyProcessingResult(
                success=False,
                document_id=document_id,
                filename="unknown",
                error="Document not found"
            )
        
        if department_id == "discard":
            return PrivacyProcessingResult(
                success=True,
                document_id=document_id,
                filename=doc.filename,
                status="discarded"
            )
        
        # Get department and route
        try:
            dept_id = Department(department_id)
            dept_config = get_department_by_id(dept_id)
            
            if not dept_config:
                return PrivacyProcessingResult(
                    success=False,
                    document_id=document_id,
                    filename=doc.filename,
                    error=f"Invalid department: {department_id}"
                )
            
            # Create a manual classification result (multilabel compatible)
            manual_classification = ClassificationResult(
                categories=[DocumentCategory(department_id)],
                confidence=1.0,  # Human decision = 100% confidence
                all_scores={department_id: 1.0},
                model_used="human_review"
            )
            
            # Forward to department
            email_sent = await self.email_forwarder.forward_document(
                to_email=dept_config.email,
                document_metadata=doc,
                classification=manual_classification
            )
            
            # Record routing
            routing_info = RoutingInfo(
                department_id=dept_id.value,
                department_name=dept_config.name,
                department_email=dept_config.email,
                delivery_status="sent" if email_sent else "failed"
            )
            
            await mongodb.update_routing(document_id, routing_info)
            
            return PrivacyProcessingResult(
                success=True,
                document_id=document_id,
                filename=doc.filename,
                categories=[department_id],
                category=department_id,
                confidence=1.0,
                routed_to=[dept_config.name],
                department_emails=[dept_config.email],
                status="routed"
            )
            
        except ValueError:
            return PrivacyProcessingResult(
                success=False,
                document_id=document_id,
                filename=doc.filename,
                error=f"Invalid department ID: {department_id}"
            )


# ============================================================
# Global Instance & Helper Functions
# ============================================================

_orchestrator: Optional[PrivacyOrchestrator] = None


def get_privacy_orchestrator() -> PrivacyOrchestrator:
    """Get or create the privacy orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PrivacyOrchestrator()
    return _orchestrator


async def process_document_privacy_first(
    document_id: str
) -> PrivacyProcessingResult:
    """
    Process a document using the privacy-first workflow.
    
    This is the main entry point for processing documents.
    """
    orchestrator = get_privacy_orchestrator()
    return await orchestrator.process_document(document_id)


async def complete_human_review(
    document_id: str,
    reviewed_by: str,
    department_id: str,
    notes: Optional[str] = None
) -> PrivacyProcessingResult:
    """Complete a human review and route the document."""
    orchestrator = get_privacy_orchestrator()
    return await orchestrator.process_human_review_decision(
        document_id=document_id,
        reviewed_by=reviewed_by,
        department_id=department_id,
        notes=notes
    )


# ============================================================
# Integration with Telegram/Email Services
# ============================================================

async def on_document_received(document_id: str) -> None:
    """
    Callback for when a new document is received and stored.
    
    This is called by Telegram/Email services after storing metadata.
    """
    print(f"🔄 Processing document: {document_id}")
    
    result = await process_document_privacy_first(document_id)
    
    if result.success:
        if result.needs_review:
            print(f"👤 Document queued for review: {result.filename}")
        else:
            print(f"✅ Document routed: {result.filename} → {result.routed_to}")
    else:
        print(f"❌ Processing failed: {result.error}")
    
    # Send notification back to source (Telegram/Email)
    await _send_source_notification(document_id, result)


async def _send_source_notification(
    document_id: str, 
    result: PrivacyProcessingResult
) -> None:
    """Send notification back to the document source."""
    mongodb = await get_mongodb()
    doc = await mongodb.get_document(document_id)
    
    if not doc:
        return
    
    # If from Telegram, send Telegram notification
    if doc.origin.source.value == "telegram" and doc.origin.chat_id:
        try:
            from services.privacy_telegram_service import get_privacy_telegram_service
            
            service = get_privacy_telegram_service()
            chat_id = int(doc.origin.chat_id)
            
            if result.needs_review:
                await service.send_human_review_notification(
                    chat_id=chat_id,
                    filename=doc.filename,
                    reason=result.review_reason or "Uncertain classification"
                )
            elif result.routed_to:
                await service.send_routing_complete(
                    chat_id=chat_id,
                    filename=doc.filename,
                    departments=result.routed_to
                )
        except Exception as e:
            print(f"⚠️ Failed to send Telegram notification: {e}")
