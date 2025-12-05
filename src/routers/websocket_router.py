"""
WebSocket Router for Real-time Document Updates

Provides WebSocket endpoints for clients to receive real-time notifications
about document changes, actions, and status updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Optional
import json

from services.websocket_service import (
    manager,
    DocumentNotification
)
from services.auth_service import decode_access_token

router = APIRouter(prefix="/ws", tags=["websocket"])


async def get_user_from_token(token: str) -> Optional[dict]:
    """
    Validate a JWT token and extract user info.
    Returns None if token is invalid.
    """
    try:
        token_data = decode_access_token(token)
        if not token_data:
            return None
        return {
            "user_id": token_data.user_id,
            "teams": token_data.teams,
            "name": token_data.email  # Use email as name fallback
        }
    except Exception:
        return None


@router.websocket("/documents")
async def websocket_documents(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time document updates.
    
    Connection URL: ws://host/ws/documents?token=<jwt_token>
    
    The token is used to authenticate and determine which teams
    the user belongs to. Updates are sent for documents in those teams.
    
    Message format (received by client):
    {
        "event_type": "document_created" | "action_performed" | "status_changed" | "priority_changed",
        "document_id": "...",
        "team": "...",
        "title": "...",
        "categories": ["..."],
        "action_type": "...",
        "performed_by": "...",
        "new_status": "...",
        "priority": "...",
        "timestamp": "..."
    }
    
    Client can also send messages:
    - {"type": "ping"} -> server responds with {"type": "pong"}
    - {"type": "subscribe", "teams": ["team1", "team2"]} -> subscribe to additional teams
    - {"type": "unsubscribe", "teams": ["team1"]} -> unsubscribe from teams
    """
    # Authenticate user
    user_info = None
    teams = []
    
    if token:
        user_info = await get_user_from_token(token)
        if user_info:
            teams = user_info.get("teams", [])
            
    # If no valid token or no teams, allow connection but with no team subscriptions
    # The client can authenticate later via message
    if not teams:
        teams = []  # Will only receive global broadcasts if admin
        
    try:
        await manager.connect(
            websocket,
            teams=teams,
            user_id=user_info.get("user_id") if user_info else None,
            user_name=user_info.get("name") if user_info else None
        )
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "teams": teams,
            "message": "Connected to document updates"
        }))
        
        # Listen for client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                msg_type = message.get("type")
                
                if msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
                elif msg_type == "authenticate":
                    # Allow late authentication
                    new_token = message.get("token")
                    if new_token:
                        new_user_info = await get_user_from_token(new_token)
                        if new_user_info:
                            # Disconnect and reconnect with new teams
                            await manager.disconnect(websocket)
                            teams = new_user_info.get("teams", [])
                            await manager.connect(
                                websocket,
                                teams=teams,
                                user_id=new_user_info.get("user_id"),
                                user_name=new_user_info.get("name")
                            )
                            await websocket.send_text(json.dumps({
                                "type": "authenticated",
                                "teams": teams
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Invalid token"
                            }))
                            
                elif msg_type == "subscribe":
                    # Subscribe to additional teams (if authorized)
                    # In production, should validate user has access to these teams
                    new_teams = message.get("teams", [])
                    current_info = manager.connection_info.get(websocket, {})
                    current_teams = set(current_info.get("teams", []))
                    current_teams.update(new_teams)
                    
                    # Re-register with updated teams
                    await manager.disconnect(websocket)
                    await manager.connect(
                        websocket,
                        teams=list(current_teams),
                        user_id=current_info.get("user_id"),
                        user_name=current_info.get("user_name")
                    )
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "teams": list(current_teams)
                    }))
                    
                elif msg_type == "unsubscribe":
                    # Unsubscribe from teams
                    remove_teams = set(message.get("teams", []))
                    current_info = manager.connection_info.get(websocket, {})
                    current_teams = set(current_info.get("teams", []))
                    current_teams -= remove_teams
                    
                    # Re-register with updated teams
                    await manager.disconnect(websocket)
                    await manager.connect(
                        websocket,
                        teams=list(current_teams),
                        user_id=current_info.get("user_id"),
                        user_name=current_info.get("user_name")
                    )
                    await websocket.send_text(json.dumps({
                        "type": "unsubscribed",
                        "teams": list(current_teams)
                    }))
                    
                elif msg_type == "status":
                    # Send connection status
                    current_info = manager.connection_info.get(websocket, {})
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "teams": current_info.get("teams", []),
                        "connected_at": current_info.get("connected_at"),
                        "user_id": current_info.get("user_id")
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@router.get("/stats")
async def get_websocket_stats():
    """
    Get current WebSocket connection statistics.
    
    Returns:
        Connection stats including total connections and per-team counts
    """
    return manager.get_connection_stats()
