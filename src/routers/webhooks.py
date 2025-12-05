"""
Webhook Router for External Integrations.

This router handles incoming webhooks from:
- Telegram (document uploads)
- Future: Gmail, Slack, etc.

These webhooks trigger the document processing pipeline.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Response
from pydantic import BaseModel, Field
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.telegram_service import (
    handle_telegram_webhook,
    get_telegram_status,
    TelegramServiceStatus,
)
from agents.orchestrator import process_document_with_agents, ProcessingResult


router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# ============================================================
# Request/Response Models
# ============================================================

class WebhookResponse(BaseModel):
    """Standard webhook response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ManualProcessRequest(BaseModel):
    """Request to manually trigger document processing."""
    filename: str = Field(..., description="Name of the document")
    content: str = Field(..., description="Text content of the document")
    source: str = Field(default="manual", description="Source of the document")


class ProcessingStatusResponse(BaseModel):
    """Response with processing status."""
    telegram: TelegramServiceStatus
    processing_enabled: bool
    message: str


# ============================================================
# Telegram Webhook Endpoints
# ============================================================

@router.post("/telegram", response_model=WebhookResponse)
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming Telegram webhook updates.
    
    This endpoint receives updates from Telegram when:
    - Users send documents to the bot
    - Users send commands to the bot
    
    The document processing happens in the background.
    """
    try:
        # Parse the update
        update_data = await request.json()
        
        # Process in background
        background_tasks.add_task(handle_telegram_webhook, update_data)
        
        return WebhookResponse(
            success=True,
            message="Webhook received and queued for processing",
            data={"update_id": update_data.get("update_id")}
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telegram/status", response_model=TelegramServiceStatus)
async def telegram_status():
    """
    Get the current status of Telegram integration.
    
    Returns:
        TelegramServiceStatus with connection and configuration details.
    """
    return get_telegram_status()


@router.post("/telegram/setup")
async def setup_telegram_webhook(webhook_url: str):
    """
    Set up the Telegram webhook URL.
    
    Args:
        webhook_url: The HTTPS URL where Telegram should send updates.
        
    Note: This is a placeholder. In production, this would configure
        the webhook with Telegram's API.
    """
    return WebhookResponse(
        success=True,
        message=f"Webhook setup initiated for: {webhook_url}",
        data={"note": "This is a placeholder - configure TELEGRAM_BOT_TOKEN to enable"}
    )


# ============================================================
# Manual Processing Endpoint
# ============================================================

@router.post("/process", response_model=ProcessingResult)
async def manual_process_document(request: ManualProcessRequest):
    """
    Manually trigger document processing.
    
    This endpoint allows testing the multi-agent workflow without
    external integrations like Telegram.
    
    Args:
        request: ManualProcessRequest with filename and content
    
    Returns:
        ProcessingResult with full processing details
    """
    try:
        result = await process_document_with_agents(
            filename=request.filename,
            text_content=request.content
        )
        return result
        
    except Exception as e:
        return ProcessingResult(
            success=False,
            filename=request.filename,
            error=str(e)
        )


# ============================================================
# Integration Status
# ============================================================

@router.get("/status", response_model=ProcessingStatusResponse)
async def get_integration_status():
    """
    Get the status of all external integrations.
    
    Returns:
        Status of Telegram and other integrations.
    """
    telegram_status = get_telegram_status()
    
    return ProcessingStatusResponse(
        telegram=telegram_status,
        processing_enabled=True,
        message="IDP webhook service is running"
    )


# ============================================================
# Future Integration Placeholders
# ============================================================

@router.post("/gmail", response_model=WebhookResponse)
async def gmail_webhook(request: Request):
    """
    Handle incoming Gmail push notifications.
    
    Note: This is a placeholder for future implementation.
    Would require:
    - Google Cloud Pub/Sub subscription
    - Gmail API OAuth2 setup
    - Watch request on user's mailbox
    """
    return WebhookResponse(
        success=False,
        message="Gmail integration not yet implemented",
        data={"status": "placeholder"}
    )


@router.post("/slack", response_model=WebhookResponse)
async def slack_webhook(request: Request):
    """
    Handle incoming Slack events.
    
    Note: This is a placeholder for future implementation.
    Would require:
    - Slack App configuration
    - Event subscriptions
    - OAuth2 setup
    """
    return WebhookResponse(
        success=False,
        message="Slack integration not yet implemented",
        data={"status": "placeholder"}
    )


# ============================================================
# API Key Management
# ============================================================

class AddApiKeyRequest(BaseModel):
    """Request to add a new API key."""
    api_key: str = Field(..., description="The Gemini API key to add")


@router.get("/llm/status")
async def llm_status():
    """
    Get status of all Gemini API keys.
    
    Returns information about:
    - Total keys configured
    - Available keys
    - Exhausted keys with cooldown times
    - Request counts per key
    """
    from services.llm_provider import get_key_status
    return get_key_status()


@router.post("/llm/add-key", response_model=WebhookResponse)
async def add_api_key(request: AddApiKeyRequest):
    """
    Add a new Gemini API key dynamically.
    
    The key will be added to the rotation pool immediately.
    """
    from services.llm_provider import add_api_key
    
    if len(request.api_key) < 20:
        raise HTTPException(status_code=400, detail="Invalid API key format")
    
    add_api_key(request.api_key)
    
    return WebhookResponse(
        success=True,
        message=f"API key added successfully (ending in ...{request.api_key[-8:]})",
        data={"key_suffix": f"...{request.api_key[-8:]}"}
    )


# ============================================================
# Health Check
# ============================================================

@router.get("/health")
async def webhook_health():
    """Health check for webhook service."""
    return {"status": "healthy", "service": "webhooks"}
