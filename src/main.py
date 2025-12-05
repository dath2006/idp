from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import routers
from routers import users, webhooks
from routers import privacy_documents  # Privacy-first router (DEFAULT)
from routers import team_documents  # Team-based document access
from routers import websocket_router  # Real-time WebSocket updates
# Legacy router - comment out to disable LLM-based processing
# from routers import documents as legacy_documents


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./app.db"
    SECRET_KEY: str = "dev-secret-key"
    APP_NAME: str = "IDP - Privacy-First Document Routing"
    
    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "idp_privacy"
    
    # Optional API keys
    GOOGLE_API_KEY: str = ""
    SENDGRID_API_KEY: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    
    # Email (IMAP)
    IMAP_SERVER: str = "imap.gmail.com"
    IMAP_EMAIL: str = ""
    IMAP_PASSWORD: str = ""
    
    # Privacy mode
    PRIVACY_MODE: bool = True  # Set to False to enable LLM processing

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Background task handles
_email_polling_task = None
_telegram_polling_task = None


# ============================================================
# Application Lifespan (startup/shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _email_polling_task, _telegram_polling_task
    import asyncio
    
    # Startup
    print("🚀 Starting IDP Privacy-First System...")
    
    # Connect to MongoDB
    try:
        from services.mongodb_service import mongodb_service
        await mongodb_service.connect()
    except Exception as e:
        print(f"⚠️ MongoDB connection failed: {e}")
    
    # Initialize Telegram bot (optional)
    if settings.TELEGRAM_BOT_TOKEN:
        try:
            from services.privacy_telegram_service import (
                get_privacy_telegram_service,
            )
            from agents.privacy_orchestrator import on_document_received as process_callback
            
            service = get_privacy_telegram_service()
            service.set_document_callback(process_callback)
            await service.initialize()
            
            # Start Telegram polling in background (for development)
            # In production, use webhook instead
            if not settings.TELEGRAM_BOT_TOKEN.startswith("WEBHOOK:"):
                _telegram_polling_task = asyncio.create_task(
                    _run_telegram_polling()
                )
                print("🤖 Telegram bot polling started")
            else:
                print("🤖 Telegram bot initialized (webhook mode)")
        except Exception as e:
            print(f"⚠️ Telegram initialization failed: {e}")
    
    # Start Email polling (optional)
    if settings.IMAP_EMAIL and settings.IMAP_PASSWORD:
        try:
            _email_polling_task = asyncio.create_task(
                _run_email_polling()
            )
            print("📧 Email polling started")
        except Exception as e:
            print(f"⚠️ Email polling failed to start: {e}")
    
    # Preload ML model (optional, speeds up first request)
    try:
        from ml.distilbert_classifier import preload_model
        preload_model()
    except Exception as e:
        print(f"⚠️ ML model preload skipped: {e}")
    
    print("✅ System ready!")
    
    yield  # Application runs here
    
    # Shutdown
    print("🛑 Shutting down...")
    
    # Cancel background tasks
    if _email_polling_task:
        _email_polling_task.cancel()
        try:
            await _email_polling_task
        except asyncio.CancelledError:
            pass
    
    if _telegram_polling_task:
        _telegram_polling_task.cancel()
        try:
            await _telegram_polling_task
        except asyncio.CancelledError:
            pass
    
    # Disconnect MongoDB
    try:
        from services.mongodb_service import mongodb_service
        await mongodb_service.disconnect()
    except Exception:
        pass
    
    # Stop Telegram
    try:
        from services.privacy_telegram_service import stop_telegram_polling
        await stop_telegram_polling()
    except Exception:
        pass
    
    print("👋 Goodbye!")


# ============================================================
# Background Task Runners
# ============================================================

async def _run_email_polling():
    """Background task for email polling."""
    from services.email_receiver_service import get_email_service
    from agents.privacy_orchestrator import on_document_received
    
    service = get_email_service()
    await service.poll_loop(on_document_received=on_document_received)


async def _run_telegram_polling():
    """Background task for Telegram polling."""
    from services.privacy_telegram_service import get_privacy_telegram_service
    
    service = get_privacy_telegram_service()
    await service.start_polling()


app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Privacy-First Document Routing for Infrastructure Operations

This API provides **compliance-safe** document routing WITHOUT reading document content.

### 🔐 Privacy Guarantee
- Documents are stored securely but **NEVER read by AI**
- Classification based on **email body/message only**
- Uses DistilBERT ML model (not LLM) for privacy-safe classification
- Full audit trail in MongoDB

### Features
- **Email/Telegram Ingestion**: Receive documents from email or Telegram
- **Privacy-Safe Classification**: Classify based on message text only
- **Human-in-the-Loop**: Uncertain documents go to review queue
- **Department Routing**: Forward to appropriate teams via email

### Architecture
1. **Document Received** → Stored to disk, metadata to MongoDB
2. **DistilBERT Classification** → Based on email/Telegram message only
3. **Routing Decision** → High confidence → auto-route, Low → human review
4. **Email Forwarding** → Document reference sent to department

### API Sections
- **/api/v1/privacy**: Privacy-first document processing (DEFAULT)
- **/api/v1/privacy/review-queue**: Human review interface
- **/api/v1/webhooks**: Telegram/Email webhooks
- **/api/v1/users**: User management

### Legacy (Optional)
The LLM-based processing is available at /api/v1/documents-legacy but is
disabled by default for compliance safety.
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Include Routers
# ============================================================

# Privacy-First Router (DEFAULT)
app.include_router(
    privacy_documents.router, 
    prefix="/api/v1/privacy", 
    tags=["Privacy Documents"]
)

# User management
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])

# Team-based document access
app.include_router(
    team_documents.router, 
    prefix="/api/v1/teams", 
    tags=["Team Documents"]
)

# WebSocket for real-time updates
app.include_router(websocket_router.router, tags=["WebSocket"])

# Webhooks (Telegram, Email) - routes to privacy system
app.include_router(webhooks.router, prefix="/api/v1", tags=["Webhooks"])

# Legacy LLM-based processing (disabled by default)
# Uncomment to enable alongside privacy system
# app.include_router(
#     legacy_documents.router, 
#     prefix="/api/v1/documents-legacy", 
#     tags=["Documents (Legacy LLM)"]
# )


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "IDP Privacy-First API is running",
        "version": "3.0.0",
        "mode": "privacy-first",
        "docs": "/docs",
        "features": [
            "Privacy-Safe Multilabel Classification (TF-IDF)",
            "Email/Telegram Document Intake",
            "Human-in-the-Loop Review",
            "Department Routing via Email",
            "MongoDB Metadata Storage",
            "No LLM Access to Document Content",
            "Team-Based Document Access",
            "Real-time WebSocket Updates",
            "JWT Authentication"
        ],
        "endpoints": {
            "privacy": "/api/v1/privacy",
            "review_queue": "/api/v1/privacy/review-queue",
            "teams": "/api/v1/teams/{team}/documents",
            "users": "/api/v1/users",
            "webhooks": "/api/v1/webhooks",
            "websocket": "/ws/documents",
            "health": "/api/v1/privacy/health"
        }
    }


@app.get("/health")
def health_check():
    """Quick health check endpoint."""
    return {
        "status": "healthy",
        "mode": "privacy-first",
        "services": {
            "api": True,
            "mongodb": os.getenv("MONGODB_URI", "").startswith("mongodb"),
            "telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
            "email": bool(os.getenv("IMAP_EMAIL")),
        },
        "note": "For detailed health, use /api/v1/privacy/health"
    }

