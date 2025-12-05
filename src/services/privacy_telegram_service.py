"""
Privacy-First Telegram Service - Full implementation for document routing.

This service handles Telegram bot interactions for the privacy-first workflow:
- Receives documents via Telegram bot
- Downloads files to local storage (NO content processing)
- Stores metadata in MongoDB
- Triggers classification based on message/caption only
- Forwards to appropriate departments

Key Features:
- Complete file downloading and storage
- Origin tracking (chat, user, group)
- Caption/message extraction for classification
- Status notifications back to users
"""

import os
import asyncio
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass

from telegram import Update, Bot
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    filters,
    ContextTypes,
)
from telegram.error import TelegramError

from services.mongodb_service import (
    DocumentMetadata,
    FileOrigin,
    DocumentSource,
    DocumentStatus,
    get_mongodb,
)


def _escape_markdown(text: str) -> str:
    """Escape special Markdown characters for Telegram."""
    if not text:
        return ""
    # Escape these characters: _ * [ ] ( ) ~ ` > # + - = | { } . !
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    return text


# ============================================================
# Configuration
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")
TELEGRAM_STORAGE_PATH = os.getenv(
    "TELEGRAM_STORAGE_PATH", 
    "./storage/telegram_files"
)

# Maximum file size (20MB - Telegram's standard limit)
MAX_FILE_SIZE = int(os.getenv("TELEGRAM_MAX_FILE_SIZE", str(20 * 1024 * 1024)))


# ============================================================
# Data Classes
# ============================================================

@dataclass
class TelegramFileInfo:
    """Information about a file received from Telegram."""
    file_id: str
    file_unique_id: str
    filename: str
    file_size: int
    mime_type: Optional[str]
    
    # Sender info
    sender_id: int
    sender_username: Optional[str]
    sender_first_name: Optional[str]
    
    # Chat info
    chat_id: int
    chat_type: str  # private, group, supergroup, channel
    chat_title: Optional[str]  # For groups
    
    # Message info
    message_id: int
    caption: Optional[str]  # The caption/message for classification
    received_at: datetime


# ============================================================
# Privacy-First Telegram Service
# ============================================================

class PrivacyTelegramService:
    """
    Full Telegram bot implementation for privacy-first document routing.
    
    This service:
    1. Receives documents from Telegram
    2. Downloads to local storage
    3. Stores ONLY metadata in MongoDB
    4. Triggers classification via callback
    5. Sends status notifications
    """
    
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.webhook_url = TELEGRAM_WEBHOOK_URL
        self.storage_path = Path(TELEGRAM_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self._is_initialized = False
        
        # Callback for when a document is received and stored
        self._on_document_stored: Optional[
            Callable[[str], Awaitable[None]]
        ] = None
    
    def set_document_callback(
        self, 
        callback: Callable[[str], Awaitable[None]]
    ) -> None:
        """
        Set callback for when a document is stored.
        
        Args:
            callback: Async function that receives document_id
        """
        self._on_document_stored = callback
    
    async def initialize(self) -> bool:
        """Initialize the Telegram bot."""
        if not self.bot_token:
            print("⚠️ TELEGRAM_BOT_TOKEN not configured")
            return False
        
        try:
            # Build application
            self.application = (
                Application.builder()
                .token(self.bot_token)
                .build()
            )
            self.bot = self.application.bot
            
            # Register handlers
            self.application.add_handler(
                CommandHandler("start", self._handle_start)
            )
            self.application.add_handler(
                CommandHandler("help", self._handle_help)
            )
            self.application.add_handler(
                CommandHandler("status", self._handle_status)
            )
            self.application.add_handler(
                MessageHandler(filters.Document.ALL, self._handle_document)
            )
            self.application.add_handler(
                MessageHandler(filters.PHOTO, self._handle_photo)
            )
            # Handle text messages (for debugging)
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
            )
            
            # Get bot info
            me = await self.bot.get_me()
            print(f"✅ Telegram bot initialized: @{me.username}")
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Telegram bot: {e}")
            return False
    
    async def start_polling(self) -> None:
        """Start polling for updates (development mode)."""
        if not self._is_initialized:
            await self.initialize()
        
        if self.application:
            print("🤖 Starting Telegram bot polling...")
            
            # Clear any existing webhook first
            try:
                await self.bot.delete_webhook(drop_pending_updates=True)
                print("🧹 Cleared existing webhook")
            except Exception as e:
                print(f"⚠️ Could not clear webhook: {e}")
            
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(
                drop_pending_updates=True,  # Ignore old messages
                allowed_updates=["message", "edited_message"]
            )
            print("✅ Telegram polling is now ACTIVE - send a message to test!")
            
            # Keep the polling task alive - but allow cancellation
            try:
                while True:
                    await asyncio.sleep(3600)  # Sleep for 1 hour intervals
            except asyncio.CancelledError:
                print("🛑 Telegram polling task cancelled")
                raise
    
    async def stop_polling(self) -> None:
        """Stop the polling."""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            print("🛑 Telegram bot stopped")
    
    async def setup_webhook(self, url: str) -> bool:
        """Set up webhook for production."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            await self.bot.set_webhook(url=url)
            print(f"🔗 Webhook set: {url}")
            return True
        except TelegramError as e:
            print(f"❌ Failed to set webhook: {e}")
            return False
    
    async def process_webhook_update(self, update_data: Dict[str, Any]) -> bool:
        """Process a webhook update."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            update = Update.de_json(update_data, self.bot)
            await self.application.process_update(update)
            return True
        except Exception as e:
            print(f"❌ Error processing webhook update: {e}")
            return False
    
    # ============================================================
    # Command Handlers
    # ============================================================
    
    async def _handle_text(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle regular text messages (for debugging)."""
        print(f"💬 Received text message: {update.message.text[:50]}...")
        await update.message.reply_text(
            "📎 Please send a *document* (PDF, Word, Excel, etc.) with a description.\n\n"
            "I need a file attachment to process!",
            parse_mode="Markdown"
        )
    
    async def _handle_start(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        print("🚀 Received /start command")
        await update.message.reply_text(
            "🔐 *Privacy-First Document Routing*\n\n"
            "Send me any document and I'll route it to the right department.\n\n"
            "✅ *Privacy Guaranteed:*\n"
            "• Documents are stored securely\n"
            "• Content is NEVER read or processed\n"
            "• Only your message is used for classification\n\n"
            "📎 Just send a document with a brief description!",
            parse_mode="Markdown"
        )
    
    async def _handle_help(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command."""
        await update.message.reply_text(
            "📚 *How to Use*\n\n"
            "1️⃣ Send a document as an attachment\n"
            "2️⃣ Include a caption describing what it is\n"
            "3️⃣ I'll route it to the right department\n\n"
            "*Supported Files:*\n"
            "📄 PDF, Word, Excel, PowerPoint\n"
            "📐 CAD files (DWG, DXF)\n"
            "🖼️ Images (JPG, PNG)\n"
            "📊 Data files (CSV, XML)\n\n"
            "*Commands:*\n"
            "/start - Introduction\n"
            "/help - This message\n"
            "/status - Check system status",
            parse_mode="Markdown"
        )
    
    async def _handle_status(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command."""
        try:
            mongodb = await get_mongodb()
            stats = await mongodb.get_statistics()
            
            pending = stats.get("pending_review", 0)
            total = stats.get("total_documents", 0)
            
            await update.message.reply_text(
                "📊 *System Status*\n\n"
                f"📁 Total Documents: {total}\n"
                f"⏳ Pending Review: {pending}\n"
                f"✅ System: Online\n\n"
                "_Privacy mode active - no content processing_",
                parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(
                "⚠️ Could not retrieve status. Please try again later."
            )
    
    # ============================================================
    # Document Handlers
    # ============================================================
    
    async def _handle_document(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming document."""
        print(f"📄 Received document from Telegram!")
        message = update.message
        document = message.document
        
        print(f"   File: {document.file_name}, Size: {document.file_size} bytes")
        
        # Check file size
        if document.file_size > MAX_FILE_SIZE:
            await message.reply_text(
                f"⚠️ File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
            )
            return
        
        # Send processing notification
        status_msg = await message.reply_text(
            "📥 Receiving document..."
        )
        
        try:
            # Create file info
            file_info = TelegramFileInfo(
                file_id=document.file_id,
                file_unique_id=document.file_unique_id,
                filename=document.file_name or "unknown_file",
                file_size=document.file_size,
                mime_type=document.mime_type,
                sender_id=message.from_user.id,
                sender_username=message.from_user.username,
                sender_first_name=message.from_user.first_name,
                chat_id=message.chat.id,
                chat_type=message.chat.type,
                chat_title=message.chat.title,
                message_id=message.message_id,
                caption=message.caption,
                received_at=datetime.utcnow(),
            )
            
            # Download and store file
            doc_id = await self._download_and_store(file_info, context)
            
            # Update status message
            await status_msg.edit_text(
                "✅ *Document Received*\n\n"
                f"📄 {file_info.filename}\n"
                f"📝 Classification in progress...\n\n"
                "_Your document is being routed to the appropriate department._",
                parse_mode="Markdown"
            )
            
            # Trigger classification callback
            if self._on_document_stored and doc_id:
                await self._on_document_stored(doc_id)
            
        except Exception as e:
            print(f"❌ Error handling document: {e}")
            await status_msg.edit_text(
                "❌ Failed to process document. Please try again."
            )
    
    async def _handle_photo(
        self, 
        update: Update, 
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming photo."""
        message = update.message
        
        # Get largest photo
        photo = message.photo[-1]  # Largest size
        
        status_msg = await message.reply_text(
            "📥 Receiving image..."
        )
        
        try:
            # Create file info
            file_info = TelegramFileInfo(
                file_id=photo.file_id,
                file_unique_id=photo.file_unique_id,
                filename=f"photo_{photo.file_unique_id}.jpg",
                file_size=photo.file_size or 0,
                mime_type="image/jpeg",
                sender_id=message.from_user.id,
                sender_username=message.from_user.username,
                sender_first_name=message.from_user.first_name,
                chat_id=message.chat.id,
                chat_type=message.chat.type,
                chat_title=message.chat.title,
                message_id=message.message_id,
                caption=message.caption,
                received_at=datetime.utcnow(),
            )
            
            doc_id = await self._download_and_store(file_info, context)
            
            await status_msg.edit_text(
                "✅ *Image Received*\n\n"
                "_Your image is being processed._",
                parse_mode="Markdown"
            )
            
            if self._on_document_stored and doc_id:
                await self._on_document_stored(doc_id)
            
        except Exception as e:
            print(f"❌ Error handling photo: {e}")
            await status_msg.edit_text(
                "❌ Failed to process image. Please try again."
            )
    
    async def _download_and_store(
        self, 
        file_info: TelegramFileInfo,
        context: ContextTypes.DEFAULT_TYPE
    ) -> Optional[str]:
        """
        Download file from Telegram and store metadata in MongoDB.
        
        Returns document ID if successful.
        """
        try:
            # Get file from Telegram
            tg_file = await context.bot.get_file(file_info.file_id)
            
            # Create storage path
            date_folder = datetime.utcnow().strftime("%Y/%m/%d")
            save_dir = self.storage_path / date_folder / str(file_info.chat_id)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Sanitize filename
            safe_filename = "".join(
                c if c.isalnum() or c in "._-" else "_" 
                for c in file_info.filename
            )
            file_path = save_dir / safe_filename
            
            # Ensure unique filename
            counter = 1
            base_path = file_path
            while file_path.exists():
                name, ext = os.path.splitext(safe_filename)
                file_path = save_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            # Download file
            await tg_file.download_to_drive(file_path)
            
            # Calculate hash
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Get actual file size
            actual_size = file_path.stat().st_size
            
            # Get file extension
            _, ext = os.path.splitext(file_info.filename)
            
            # Create origin metadata
            origin = FileOrigin(
                source=DocumentSource.TELEGRAM,
                source_id=str(file_info.message_id),
                sender_telegram_id=str(file_info.sender_id),
                sender_telegram_username=file_info.sender_username,
                chat_id=str(file_info.chat_id),
                chat_title=file_info.chat_title,
                original_body=file_info.caption,  # Caption for classification
                received_at=file_info.received_at,
            )
            
            # Create document metadata
            doc_metadata = DocumentMetadata(
                filename=file_info.filename,
                file_extension=ext.lower(),
                file_size_bytes=actual_size,
                file_path=str(file_path),
                file_hash=file_hash,
                origin=origin,
                status=DocumentStatus.RECEIVED,
            )
            
            # Store in MongoDB
            mongodb = await get_mongodb()
            doc_id = await mongodb.create_document(doc_metadata)
            
            print(f"📱 Stored Telegram file: {file_info.filename} (ID: {doc_id})")
            
            return doc_id
            
        except Exception as e:
            print(f"❌ Error downloading/storing file: {e}")
            raise
    
    # ============================================================
    # Notification Methods
    # ============================================================
    
    async def send_classification_result(
        self,
        chat_id: int,
        filename: str,
        category: str,
        department: str,
        confidence: float,
        needs_review: bool = False
    ) -> bool:
        """Send classification result to user."""
        try:
            # Escape special characters for MarkdownV2
            safe_filename = _escape_markdown(filename)
            safe_category = _escape_markdown(category)
            safe_department = _escape_markdown(department)
            confidence_str = f"{confidence:.0%}".replace("%", "\\%")
            
            if needs_review:
                text = (
                    "🔍 *Document Needs Review*\n\n"
                    f"📄 {safe_filename}\n"
                    f"❓ Category: Uncertain\n\n"
                    "_A human reviewer will route this document\\._"
                )
            else:
                text = (
                    "✅ *Document Routed*\n\n"
                    f"📄 {safe_filename}\n"
                    f"🏷️ Category: {safe_category}\n"
                    f"🏢 Department: {safe_department}\n"
                    f"📊 Confidence: {confidence_str}\n\n"
                    "_Document has been forwarded\\._"
                )
            
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="MarkdownV2"
            )
            return True
            
        except TelegramError as e:
            print(f"❌ Failed to send notification: {e}")
            # Fallback without markdown
            try:
                if needs_review:
                    text = f"🔍 Document Needs Review\n\n📄 {filename}\n❓ Category: Uncertain\n\nA human reviewer will route this document."
                else:
                    text = f"✅ Document Routed\n\n📄 {filename}\n🏷️ Category: {category}\n🏢 Department: {department}\n📊 Confidence: {confidence:.0%}\n\nDocument has been forwarded."
                await self.bot.send_message(chat_id=chat_id, text=text)
                return True
            except:
                return False
    
    async def send_human_review_notification(
        self,
        chat_id: int,
        filename: str,
        reason: str
    ) -> bool:
        """Notify user that document needs human review."""
        try:
            # Escape special characters to avoid Markdown parsing errors
            safe_filename = _escape_markdown(filename)
            safe_reason = _escape_markdown(reason)
            await self.bot.send_message(
                chat_id=chat_id,
                text=(
                    "👤 *Human Review Required*\n\n"
                    f"📄 {safe_filename}\n"
                    f"📋 Reason: {safe_reason}\n\n"
                    "_A team member will review and route your document\._"
                ),
                parse_mode="MarkdownV2"
            )
            return True
        except TelegramError as e:
            print(f"❌ Failed to send review notification: {e}")
            # Fallback: try without markdown
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "👤 Human Review Required\n\n"
                        f"📄 {filename}\n"
                        f"📋 Reason: {reason}\n\n"
                        "A team member will review and route your document."
                    )
                )
                return True
            except:
                return False
    
    async def send_routing_complete(
        self,
        chat_id: int,
        filename: str,
        departments: List[str]
    ) -> bool:
        """Notify user that document has been routed."""
        try:
            dept_list = ", ".join(departments)
            # Escape special characters for MarkdownV2
            safe_filename = _escape_markdown(filename)
            safe_dept_list = _escape_markdown(dept_list)
            await self.bot.send_message(
                chat_id=chat_id,
                text=(
                    "📨 *Document Delivered*\n\n"
                    f"📄 {safe_filename}\n"
                    f"🏢 Sent to: {safe_dept_list}\n\n"
                    "✅ _Routing complete\\!_"
                ),
                parse_mode="MarkdownV2"
            )
            return True
        except TelegramError as e:
            print(f"❌ Failed to send completion notification: {e}")
            # Fallback: try without markdown
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "📨 Document Delivered\n\n"
                        f"📄 {filename}\n"
                        f"🏢 Sent to: {dept_list}\n\n"
                        "✅ Routing complete!"
                    )
                )
                return True
            except:
                return False


# ============================================================
# Global Service Instance
# ============================================================

_telegram_service: Optional[PrivacyTelegramService] = None


def get_privacy_telegram_service() -> PrivacyTelegramService:
    """Get or create the Telegram service instance."""
    global _telegram_service
    if _telegram_service is None:
        _telegram_service = PrivacyTelegramService()
    return _telegram_service


async def initialize_telegram() -> bool:
    """Initialize the Telegram service."""
    service = get_privacy_telegram_service()
    return await service.initialize()


async def start_telegram_polling() -> None:
    """Start Telegram bot in polling mode."""
    service = get_privacy_telegram_service()
    await service.start_polling()


async def stop_telegram_polling() -> None:
    """Stop Telegram bot."""
    if _telegram_service:
        await _telegram_service.stop_polling()


async def process_telegram_webhook(update_data: Dict[str, Any]) -> bool:
    """Process a Telegram webhook update."""
    service = get_privacy_telegram_service()
    return await service.process_webhook_update(update_data)


# ============================================================
# Health Check
# ============================================================

async def check_telegram_connection() -> Dict[str, Any]:
    """Check Telegram bot connection."""
    if not TELEGRAM_BOT_TOKEN:
        return {
            "status": "not_configured",
            "message": "TELEGRAM_BOT_TOKEN not set"
        }
    
    try:
        service = get_privacy_telegram_service()
        if not service._is_initialized:
            await service.initialize()
        
        me = await service.bot.get_me()
        
        return {
            "status": "connected",
            "bot_username": me.username,
            "bot_id": me.id,
            "webhook_configured": bool(TELEGRAM_WEBHOOK_URL)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
