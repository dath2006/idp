"""
Email Notification Service using SendGrid.

This service handles sending email notifications to users and departments
when documents are processed and routed.
"""

import os
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# SendGrid import - will be used when API key is configured
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False


class NotificationType(str, Enum):
    """Types of notifications."""
    DOCUMENT_RECEIVED = "document_received"
    DOCUMENT_PROCESSED = "document_processed"
    DOCUMENT_ROUTED = "document_routed"
    ACTION_REQUIRED = "action_required"
    SYSTEM_ALERT = "system_alert"


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationRequest(BaseModel):
    """Request to send a notification."""
    recipient_emails: List[str] = Field(..., description="List of recipient email addresses")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content (can be HTML)")
    notification_type: NotificationType = Field(default=NotificationType.DOCUMENT_PROCESSED)
    priority: NotificationPriority = Field(default=NotificationPriority.NORMAL)
    document_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document-related metadata")
    is_html: bool = Field(default=True, description="Whether body is HTML")


class NotificationResult(BaseModel):
    """Result of a notification attempt."""
    success: bool
    message: str
    notification_id: Optional[str] = None
    failed_recipients: List[str] = Field(default_factory=list)


class NotificationService:
    """Service for sending email notifications via SendGrid."""
    
    def __init__(self):
        self.api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("SENDGRID_FROM_EMAIL", "idp-system@company.com")
        self.from_name = os.getenv("SENDGRID_FROM_NAME", "IDP System")
        self.enabled = bool(self.api_key) and SENDGRID_AVAILABLE
        
        if self.enabled:
            self.client = SendGridAPIClient(self.api_key)
        else:
            self.client = None
    
    def _build_document_notification_html(
        self,
        document_name: str,
        document_type: str,
        departments: List[str],
        summary: str,
        extracted_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build HTML email body for document notification."""
        departments_html = "".join([f"<li>{dept}</li>" for dept in departments])
        
        extracted_section = ""
        if extracted_data:
            items = "".join([f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" 
                           for k, v in extracted_data.items()])
            extracted_section = f"""
            <h3>Extracted Information</h3>
            <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
                {items}
            </table>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .metadata {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .departments {{ margin: 10px 0; }}
                table {{ width: 100%; margin-top: 15px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.9em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📄 Document Processed</h1>
            </div>
            <div class="content">
                <div class="metadata">
                    <p><strong>Document:</strong> {document_name}</p>
                    <p><strong>Type:</strong> {document_type}</p>
                    <div class="departments">
                        <strong>Routed to:</strong>
                        <ul>{departments_html}</ul>
                    </div>
                </div>
                
                <h3>Summary</h3>
                <p>{summary}</p>
                
                {extracted_section}
                
                <div class="footer">
                    <p>This is an automated notification from the Intelligent Document Processing System.</p>
                    <p>Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    async def send_notification(self, request: NotificationRequest) -> NotificationResult:
        """
        Send an email notification.
        
        Args:
            request: NotificationRequest with email details
        
        Returns:
            NotificationResult with success status
        """
        if not self.enabled:
            # Log notification instead of sending when not configured
            print(f"[NOTIFICATION - {request.notification_type.value}]")
            print(f"  To: {', '.join(request.recipient_emails)}")
            print(f"  Subject: {request.subject}")
            print(f"  Priority: {request.priority.value}")
            return NotificationResult(
                success=True,
                message="Notification logged (SendGrid not configured)",
                notification_id="mock-" + str(hash(request.subject))[:8]
            )
        
        try:
            content_type = "text/html" if request.is_html else "text/plain"
            
            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=[To(email) for email in request.recipient_emails],
                subject=request.subject,
                html_content=request.body if request.is_html else None,
                plain_text_content=request.body if not request.is_html else None
            )
            
            # Add priority header
            if request.priority == NotificationPriority.URGENT:
                message.header = {"X-Priority": "1"}
            elif request.priority == NotificationPriority.HIGH:
                message.header = {"X-Priority": "2"}
            
            response = self.client.send(message)
            
            return NotificationResult(
                success=response.status_code in [200, 201, 202],
                message=f"Email sent successfully (status: {response.status_code})",
                notification_id=response.headers.get("X-Message-Id")
            )
            
        except Exception as e:
            return NotificationResult(
                success=False,
                message=f"Failed to send email: {str(e)}",
                failed_recipients=request.recipient_emails
            )
    
    async def notify_document_processed(
        self,
        document_name: str,
        document_type: str,
        departments: List[str],
        department_emails: List[str],
        summary: str,
        extracted_data: Optional[Dict[str, Any]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> NotificationResult:
        """
        Send notification about a processed document.
        
        Args:
            document_name: Name of the processed document
            document_type: Type/category of the document
            departments: List of department names document was routed to
            department_emails: Email addresses to notify
            summary: AI-generated summary of the document
            extracted_data: Structured data extracted from document
            priority: Notification priority level
        
        Returns:
            NotificationResult
        """
        html_body = self._build_document_notification_html(
            document_name=document_name,
            document_type=document_type,
            departments=departments,
            summary=summary,
            extracted_data=extracted_data
        )
        
        request = NotificationRequest(
            recipient_emails=department_emails,
            subject=f"[IDP] New Document: {document_name}",
            body=html_body,
            notification_type=NotificationType.DOCUMENT_ROUTED,
            priority=priority,
            document_metadata={
                "document_name": document_name,
                "document_type": document_type,
                "departments": departments
            },
            is_html=True
        )
        
        return await self.send_notification(request)
    
    async def notify_action_required(
        self,
        recipient_emails: List[str],
        document_name: str,
        action_description: str,
        deadline: Optional[str] = None
    ) -> NotificationResult:
        """
        Send notification requiring user action.
        
        Args:
            recipient_emails: Email addresses to notify
            document_name: Name of the document requiring action
            action_description: Description of required action
            deadline: Optional deadline for action
        
        Returns:
            NotificationResult
        """
        deadline_text = f"<p><strong>Deadline:</strong> {deadline}</p>" if deadline else ""
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: #e74c3c; color: white; padding: 20px;">
                <h1>⚠️ Action Required</h1>
            </div>
            <div style="padding: 20px;">
                <p><strong>Document:</strong> {document_name}</p>
                <p><strong>Required Action:</strong> {action_description}</p>
                {deadline_text}
                <p style="margin-top: 20px;">Please take action as soon as possible.</p>
            </div>
        </body>
        </html>
        """
        
        request = NotificationRequest(
            recipient_emails=recipient_emails,
            subject=f"[ACTION REQUIRED] {document_name}",
            body=html_body,
            notification_type=NotificationType.ACTION_REQUIRED,
            priority=NotificationPriority.HIGH,
            is_html=True
        )
        
        return await self.send_notification(request)


# Create singleton instance
notification_service = NotificationService()


# LangChain tool for sending notifications
@tool
async def send_document_notification(
    document_name: str,
    document_type: str,
    departments: str,
    department_emails: str,
    summary: str
) -> str:
    """
    Send email notification about a processed document to relevant departments.
    
    Args:
        document_name: Name of the processed document
        document_type: Type/category of the document (e.g., "invoice", "CAD drawing")
        departments: Comma-separated list of department names
        department_emails: Comma-separated list of email addresses
        summary: Brief summary of the document content
    
    Returns:
        A message indicating success or failure of the notification
    """
    dept_list = [d.strip() for d in departments.split(",")]
    email_list = [e.strip() for e in department_emails.split(",")]
    
    result = await notification_service.notify_document_processed(
        document_name=document_name,
        document_type=document_type,
        departments=dept_list,
        department_emails=email_list,
        summary=summary
    )
    
    if result.success:
        return f"✅ Notification sent successfully to {len(email_list)} recipient(s): {result.message}"
    else:
        return f"❌ Notification failed: {result.message}"
