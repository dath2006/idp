"""
Email Receiver Service - IMAP-based email receiving for privacy-first document routing.

Uses free IMAP access (Gmail, Outlook, Yahoo, etc.) to:
- Poll for new emails with attachments
- Download attachments to local storage
- Extract metadata only (no document content processing)
- Forward to classification pipeline

Supports:
- Gmail (free IMAP access with App Password)
- Outlook/Hotmail (free IMAP access)
- Any IMAP-compatible email provider

NOTE: For Gmail, enable "Less secure app access" or use App Password.
      For Outlook, IMAP is enabled by default.
"""

import os
import asyncio
import email
import hashlib
import imaplib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.header import decode_header
from email.utils import parseaddr
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from services.mongodb_service import (
    DocumentMetadata,
    FileOrigin,
    DocumentSource,
    DocumentStatus,
    get_mongodb,
)


# ============================================================
# Configuration
# ============================================================

# Email server settings (from environment)
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
IMAP_EMAIL = os.getenv("IMAP_EMAIL", "")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD", "")  # App password for Gmail
IMAP_USE_SSL = os.getenv("IMAP_USE_SSL", "true").lower() == "true"

# Polling settings
POLL_INTERVAL_SECONDS = int(os.getenv("EMAIL_POLL_INTERVAL", "60"))
MAILBOX_FOLDER = os.getenv("IMAP_MAILBOX", "INBOX")

# File storage
ATTACHMENT_STORAGE_PATH = os.getenv(
    "ATTACHMENT_STORAGE_PATH", 
    "./storage/attachments"
)

# Maximum attachment size (50MB default)
MAX_ATTACHMENT_SIZE = int(os.getenv("MAX_ATTACHMENT_SIZE", str(50 * 1024 * 1024)))


# ============================================================
# Data Models
# ============================================================

@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    content_type: str
    size_bytes: int
    content: bytes
    file_hash: str = ""
    
    def __post_init__(self):
        if not self.file_hash:
            self.file_hash = hashlib.sha256(self.content).hexdigest()


@dataclass
class ReceivedEmail:
    """Represents a received email with attachments."""
    message_id: str
    subject: str
    sender_email: str
    sender_name: str
    body_text: str
    body_html: Optional[str] = None
    received_at: datetime = field(default_factory=datetime.utcnow)
    attachments: List[EmailAttachment] = field(default_factory=list)
    raw_headers: Dict[str, str] = field(default_factory=dict)


# ============================================================
# IMAP Email Receiver
# ============================================================

class EmailReceiverService:
    """
    IMAP-based email receiver for document intake.
    
    Polls IMAP server for new emails, downloads attachments,
    and stores metadata for classification.
    """
    
    def __init__(
        self,
        server: str = IMAP_SERVER,
        port: int = IMAP_PORT,
        email_address: str = IMAP_EMAIL,
        password: str = IMAP_PASSWORD,
        use_ssl: bool = IMAP_USE_SSL,
        storage_path: str = ATTACHMENT_STORAGE_PATH,
    ):
        self.server = server
        self.port = port
        self.email_address = email_address
        self.password = password
        self.use_ssl = use_ssl
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._connection: Optional[imaplib.IMAP4_SSL] = None
        self._is_running = False
        self._processed_ids: set = set()  # Track processed message IDs
    
    def _connect(self) -> imaplib.IMAP4_SSL:
        """Establish connection to IMAP server."""
        if self.use_ssl:
            connection = imaplib.IMAP4_SSL(self.server, self.port)
        else:
            connection = imaplib.IMAP4(self.server, self.port)
        
        connection.login(self.email_address, self.password)
        return connection
    
    def _disconnect(self) -> None:
        """Close IMAP connection."""
        if self._connection:
            try:
                self._connection.logout()
            except Exception:
                pass
            self._connection = None
    
    def _decode_header_value(self, value: str) -> str:
        """Decode email header value (handles encoded headers)."""
        if not value:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(value):
            if isinstance(part, bytes):
                decoded_parts.append(
                    part.decode(encoding or 'utf-8', errors='replace')
                )
            else:
                decoded_parts.append(part)
        
        return " ".join(decoded_parts)
    
    def _extract_body(self, msg: email.message.Message) -> Tuple[str, Optional[str]]:
        """Extract text and HTML body from email message."""
        text_body = ""
        html_body = None
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        text_body = payload.decode(charset, errors='replace')
                    except Exception:
                        pass
                
                elif content_type == "text/html" and not html_body:
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        html_body = payload.decode(charset, errors='replace')
                    except Exception:
                        pass
        else:
            # Non-multipart message
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                if msg.get_content_type() == "text/html":
                    html_body = payload.decode(charset, errors='replace')
                else:
                    text_body = payload.decode(charset, errors='replace')
            except Exception:
                pass
        
        return text_body, html_body
    
    def _extract_attachments(
        self, 
        msg: email.message.Message
    ) -> List[EmailAttachment]:
        """Extract attachments from email message."""
        attachments = []
        
        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            
            if "attachment" not in content_disposition:
                continue
            
            filename = part.get_filename()
            if not filename:
                continue
            
            filename = self._decode_header_value(filename)
            
            try:
                content = part.get_payload(decode=True)
                if content and len(content) <= MAX_ATTACHMENT_SIZE:
                    attachments.append(EmailAttachment(
                        filename=filename,
                        content_type=part.get_content_type(),
                        size_bytes=len(content),
                        content=content
                    ))
            except Exception as e:
                print(f"⚠️ Error extracting attachment {filename}: {e}")
        
        return attachments
    
    def _parse_email(
        self, 
        raw_email: bytes, 
        message_id: str
    ) -> Optional[ReceivedEmail]:
        """Parse raw email bytes into ReceivedEmail object."""
        try:
            msg = email.message_from_bytes(raw_email)
            
            # Extract headers
            subject = self._decode_header_value(msg.get("Subject", ""))
            from_header = msg.get("From", "")
            sender_name, sender_email = parseaddr(from_header)
            sender_name = self._decode_header_value(sender_name)
            
            # Extract body
            text_body, html_body = self._extract_body(msg)
            
            # Extract attachments
            attachments = self._extract_attachments(msg)
            
            return ReceivedEmail(
                message_id=message_id,
                subject=subject,
                sender_email=sender_email,
                sender_name=sender_name,
                body_text=text_body,
                body_html=html_body,
                attachments=attachments,
                raw_headers={
                    "date": msg.get("Date", ""),
                    "from": from_header,
                    "to": msg.get("To", ""),
                    "cc": msg.get("Cc", ""),
                }
            )
            
        except Exception as e:
            print(f"❌ Error parsing email {message_id}: {e}")
            return None
    
    def _save_attachment(
        self, 
        attachment: EmailAttachment,
        email_id: str
    ) -> str:
        """
        Save attachment to local storage.
        
        Returns the file path.
        """
        # Create date-based subdirectory
        date_folder = datetime.utcnow().strftime("%Y/%m/%d")
        save_dir = self.storage_path / date_folder / email_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(
            c if c.isalnum() or c in "._-" else "_" 
            for c in attachment.filename
        )
        
        # Ensure unique filename
        file_path = save_dir / safe_filename
        counter = 1
        while file_path.exists():
            name, ext = os.path.splitext(safe_filename)
            file_path = save_dir / f"{name}_{counter}{ext}"
            counter += 1
        
        # Write file
        with open(file_path, "wb") as f:
            f.write(attachment.content)
        
        return str(file_path)
    
    def _fetch_emails_sync(
        self, 
        mailbox: str = MAILBOX_FOLDER,
        mark_as_read: bool = True
    ) -> List[ReceivedEmail]:
        """
        Synchronous method to fetch emails (runs in thread executor).
        """
        emails = []
        
        try:
            # Connect to server
            self._connection = self._connect()
            self._connection.select(mailbox)
            
            # Search for unread emails
            status, message_ids = self._connection.search(None, "UNSEEN")
            
            if status != "OK" or not message_ids[0]:
                return emails
            
            for msg_id in message_ids[0].split():
                msg_id_str = msg_id.decode()
                
                # Skip already processed
                if msg_id_str in self._processed_ids:
                    continue
                
                # Fetch email
                status, msg_data = self._connection.fetch(msg_id, "(RFC822)")
                
                if status != "OK":
                    continue
                
                raw_email = msg_data[0][1]
                parsed = self._parse_email(raw_email, msg_id_str)
                
                if parsed:
                    emails.append(parsed)
                    self._processed_ids.add(msg_id_str)
                    
                    if mark_as_read:
                        self._connection.store(msg_id, "+FLAGS", "\\Seen")
            
            return emails
            
        except Exception as e:
            print(f"❌ Error fetching emails: {e}")
            return emails
        
        finally:
            self._disconnect()

    async def fetch_new_emails(
        self, 
        mailbox: str = MAILBOX_FOLDER,
        mark_as_read: bool = True
    ) -> List[ReceivedEmail]:
        """
        Fetch new unread emails from the mailbox.
        
        Runs blocking IMAP operations in a thread executor to avoid
        blocking the asyncio event loop.
        
        Returns list of ReceivedEmail objects.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Use default executor
            lambda: self._fetch_emails_sync(mailbox, mark_as_read)
        )
    
    async def process_email(
        self, 
        received_email: ReceivedEmail
    ) -> List[str]:
        """
        Process a received email and store attachments.
        
        Returns list of created document IDs.
        """
        from services.mongodb_service import get_mongodb
        
        mongodb = await get_mongodb()
        document_ids = []
        
        for attachment in received_email.attachments:
            try:
                # Save attachment to disk
                file_path = self._save_attachment(
                    attachment, 
                    received_email.message_id
                )
                
                # Get file extension
                _, ext = os.path.splitext(attachment.filename)
                
                # Create file origin metadata
                origin = FileOrigin(
                    source=DocumentSource.EMAIL,
                    source_id=received_email.message_id,
                    sender_email=received_email.sender_email,
                    original_subject=received_email.subject,
                    original_body=received_email.body_text,  # For classification
                )
                
                # Create document metadata
                doc_metadata = DocumentMetadata(
                    filename=attachment.filename,
                    file_extension=ext.lower(),
                    file_size_bytes=attachment.size_bytes,
                    file_path=file_path,
                    file_hash=attachment.file_hash,
                    origin=origin,
                    status=DocumentStatus.RECEIVED,
                )
                
                # Store in MongoDB
                doc_id = await mongodb.create_document(doc_metadata)
                document_ids.append(doc_id)
                
                print(f"📧 Stored email attachment: {attachment.filename} (ID: {doc_id})")
                
            except Exception as e:
                print(f"❌ Error processing attachment {attachment.filename}: {e}")
        
        return document_ids
    
    async def poll_loop(
        self,
        on_document_received: Optional[callable] = None,
        interval: int = POLL_INTERVAL_SECONDS
    ) -> None:
        """
        Continuous polling loop for new emails.
        
        Args:
            on_document_received: Callback for each new document (doc_id)
            interval: Seconds between polls
        """
        self._is_running = True
        print(f"📬 Starting email polling (every {interval}s)...")
        
        try:
            while self._is_running:
                try:
                    emails = await self.fetch_new_emails()
                    
                    for email_msg in emails:
                        doc_ids = await self.process_email(email_msg)
                        
                        if on_document_received:
                            for doc_id in doc_ids:
                                await on_document_received(doc_id)
                    
                    if emails:
                        print(f"📧 Processed {len(emails)} email(s)")
                    
                except asyncio.CancelledError:
                    raise  # Re-raise cancellation
                except Exception as e:
                    print(f"❌ Email poll error: {e}")
                
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            print("🛑 Email polling task cancelled")
            raise
    
    def stop_polling(self) -> None:
        """Stop the polling loop."""
        self._is_running = False
        print("🛑 Email polling stopped")


# ============================================================
# Global Service Instance
# ============================================================

_email_service: Optional[EmailReceiverService] = None


def get_email_service() -> EmailReceiverService:
    """Get or create the email receiver service."""
    global _email_service
    if _email_service is None:
        _email_service = EmailReceiverService()
    return _email_service


async def start_email_polling(
    on_document_received: Optional[callable] = None
) -> None:
    """Start the email polling background task."""
    service = get_email_service()
    await service.poll_loop(on_document_received)


def stop_email_polling() -> None:
    """Stop the email polling."""
    if _email_service:
        _email_service.stop_polling()


# ============================================================
# Health Check
# ============================================================

async def check_email_connection() -> Dict[str, Any]:
    """Check if email connection is working."""
    service = get_email_service()
    
    if not service.email_address or not service.password:
        return {
            "status": "not_configured",
            "message": "IMAP_EMAIL and IMAP_PASSWORD not set"
        }
    
    try:
        connection = service._connect()
        connection.select(MAILBOX_FOLDER)
        status, _ = connection.search(None, "ALL")
        connection.logout()
        
        return {
            "status": "connected",
            "server": service.server,
            "email": service.email_address,
            "mailbox": MAILBOX_FOLDER
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
