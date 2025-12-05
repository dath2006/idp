"""
Privacy-First Document Routing Configuration.

Central configuration for the privacy-safe document processing system.
All settings can be overridden via environment variables.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrivacySettings(BaseSettings):
    """
    Privacy-first system configuration.
    
    Load from environment variables with .env file support.
    """
    
    # ============================================================
    # MongoDB Configuration
    # ============================================================
    
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    MONGODB_DATABASE: str = Field(
        default="idp_privacy",
        description="MongoDB database name"
    )
    
    # ============================================================
    # File Storage
    # ============================================================
    
    ATTACHMENT_STORAGE_PATH: str = Field(
        default="./storage/attachments",
        description="Path for email attachments"
    )
    TELEGRAM_STORAGE_PATH: str = Field(
        default="./storage/telegram_files",
        description="Path for Telegram files"
    )
    MANUAL_UPLOAD_PATH: str = Field(
        default="./storage/manual_uploads",
        description="Path for manual uploads"
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=50,
        description="Maximum file size in MB"
    )
    
    # ============================================================
    # Telegram Bot
    # ============================================================
    
    TELEGRAM_BOT_TOKEN: str = Field(
        default="",
        description="Telegram Bot API token from @BotFather"
    )
    TELEGRAM_WEBHOOK_URL: str = Field(
        default="",
        description="Webhook URL for production (HTTPS required)"
    )
    
    # ============================================================
    # Email Receiver (IMAP)
    # ============================================================
    
    IMAP_SERVER: str = Field(
        default="imap.gmail.com",
        description="IMAP server hostname"
    )
    IMAP_PORT: int = Field(
        default=993,
        description="IMAP server port (993 for SSL)"
    )
    IMAP_EMAIL: str = Field(
        default="",
        description="Email address for receiving documents"
    )
    IMAP_PASSWORD: str = Field(
        default="",
        description="Email password or app password"
    )
    IMAP_USE_SSL: bool = Field(
        default=True,
        description="Use SSL for IMAP connection"
    )
    IMAP_MAILBOX: str = Field(
        default="INBOX",
        description="Mailbox folder to monitor"
    )
    EMAIL_POLL_INTERVAL: int = Field(
        default=60,
        description="Seconds between email polling"
    )
    
    # ============================================================
    # Email Sender (SMTP)
    # ============================================================
    
    SMTP_SERVER: str = Field(
        default="smtp.gmail.com",
        description="SMTP server hostname"
    )
    SMTP_PORT: int = Field(
        default=587,
        description="SMTP server port (587 for TLS)"
    )
    SMTP_EMAIL: str = Field(
        default="",
        description="Email address for sending notifications"
    )
    SMTP_PASSWORD: str = Field(
        default="",
        description="SMTP password or app password"
    )
    
    # SendGrid (alternative to SMTP)
    SENDGRID_API_KEY: str = Field(
        default="",
        description="SendGrid API key for email sending"
    )
    SENDGRID_FROM_EMAIL: str = Field(
        default="",
        description="SendGrid sender email address"
    )
    
    # ============================================================
    # ML Classification (DistilBERT)
    # ============================================================
    
    DISTILBERT_MODEL_PATH: str = Field(
        default="./models/document_classifier",
        description="Path to custom DistilBERT model"
    )
    USE_GPU: bool = Field(
        default=False,
        description="Use GPU for ML inference"
    )
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6,
        description="Minimum confidence for auto-routing"
    )
    CLASSIFICATION_REVIEW_THRESHOLD: float = Field(
        default=0.5,
        description="Below this, always require human review"
    )
    
    # ============================================================
    # Privacy Mode
    # ============================================================
    
    PRIVACY_MODE: bool = Field(
        default=True,
        description="Enable privacy-first mode (disable LLM processing)"
    )
    ENABLE_LEGACY_LLM: bool = Field(
        default=False,
        description="Enable legacy LLM-based processing endpoints"
    )
    
    # ============================================================
    # Application
    # ============================================================
    
    APP_NAME: str = Field(
        default="IDP Privacy-First",
        description="Application name"
    )
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        description="Secret key for sessions/tokens"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# ============================================================
# Singleton Settings Instance
# ============================================================

_settings: Optional[PrivacySettings] = None


def get_privacy_settings() -> PrivacySettings:
    """Get or create privacy settings instance."""
    global _settings
    if _settings is None:
        _settings = PrivacySettings()
    return _settings


# ============================================================
# Helper Functions
# ============================================================

def is_privacy_mode_enabled() -> bool:
    """Check if privacy mode is enabled."""
    return get_privacy_settings().PRIVACY_MODE


def is_telegram_configured() -> bool:
    """Check if Telegram is configured."""
    return bool(get_privacy_settings().TELEGRAM_BOT_TOKEN)


def is_email_configured() -> bool:
    """Check if email receiving is configured."""
    settings = get_privacy_settings()
    return bool(settings.IMAP_EMAIL and settings.IMAP_PASSWORD)


def is_mongodb_configured() -> bool:
    """Check if MongoDB is configured."""
    return get_privacy_settings().MONGODB_URI.startswith("mongodb")


def get_max_file_size_bytes() -> int:
    """Get maximum file size in bytes."""
    return get_privacy_settings().MAX_FILE_SIZE_MB * 1024 * 1024
