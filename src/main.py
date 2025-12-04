from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import users, documents, webhooks
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./app.db"
    SECRET_KEY: str = "dev-secret-key"
    APP_NAME: str = "IDP - Intelligent Document Processing"
    
    # Optional API keys
    GOOGLE_API_KEY: str = ""
    SENDGRID_API_KEY: str = ""
    TELEGRAM_BOT_TOKEN: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Intelligent Document Processing for Infrastructure Operations

This API provides automated document processing capabilities:

### Features
- **Multi-Source Ingestion**: Receive documents from Telegram, Email (future), and direct upload
- **Intelligent Classification**: Automatically classify documents by type and content
- **Structured Extraction**: Extract key data into standardized JSON format
- **Role-Based Routing**: Route documents to appropriate departments
- **Email Notifications**: Notify stakeholders when documents are processed

### Multi-Agent Architecture
The system uses a supervisor-based multi-agent workflow:
1. **Classification Agent**: Analyzes file types and document content
2. **Extraction Agent**: Extracts structured data from documents
3. **Routing Agent**: Determines department routing and sends notifications

### API Sections
- **/api/v1/documents**: Document upload and processing
- **/api/v1/webhooks**: External integrations (Telegram, Gmail)
- **/api/v1/users**: User management
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["Webhooks"])


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "IDP API is running",
        "version": "2.0.0",
        "docs": "/docs",
        "features": [
            "Document Classification",
            "Data Extraction", 
            "Department Routing",
            "Email Notifications",
            "Telegram Integration (placeholder)"
        ]
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "api": True,
            "agents": True,
            "telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
            "sendgrid": bool(os.getenv("SENDGRID_API_KEY")),
        }
    }

