"""
WebSocket Service for Real-time Document Updates

Provides real-time notifications to connected clients when documents are created,
updated, or when actions are performed on documents.
"""

import json
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import asyncio
from dataclasses import dataclass, asdict


@dataclass
class DocumentNotification:
    """Real-time document notification payload"""
    event_type: str  # document_created, document_updated, action_performed, status_changed
    document_id: str
    team: str
    title: Optional[str] = None
    categories: Optional[list] = None
    action_type: Optional[str] = None
    performed_by: Optional[str] = None
    new_status: Optional[str] = None
    priority: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_json(self) -> str:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(data)


class ConnectionManager:
    """
    Manages WebSocket connections organized by team.
    
    Each team has its own set of connections, allowing targeted broadcasts
    when documents relevant to specific teams are updated.
    """
    
    def __init__(self):
        # Map of team_name -> set of WebSocket connections
        self.team_connections: Dict[str, Set[WebSocket]] = {}
        # Map of WebSocket -> user info for tracking
        self.connection_info: Dict[WebSocket, dict] = {}
        # Global connections (admin/all updates)
        self.global_connections: Set[WebSocket] = set()
        
    async def connect(
        self, 
        websocket: WebSocket, 
        teams: list[str],
        user_id: Optional[str] = None,
        user_name: Optional[str] = None
    ):
        """
        Accept a WebSocket connection and register it for specified teams.
        
        Args:
            websocket: The WebSocket connection
            teams: List of teams this connection should receive updates for
            user_id: Optional user ID for tracking
            user_name: Optional user name for tracking
        """
        await websocket.accept()
        
        # Store connection info
        self.connection_info[websocket] = {
            "user_id": user_id,
            "user_name": user_name,
            "teams": teams,
            "connected_at": datetime.utcnow().isoformat()
        }
        
        # Register for each team
        for team in teams:
            if team not in self.team_connections:
                self.team_connections[team] = set()
            self.team_connections[team].add(websocket)
            
        # If user has "admin" role or is in all teams, add to global
        if "admin" in teams or "all" in teams:
            self.global_connections.add(websocket)
            
    async def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection from all teams.
        """
        if websocket in self.connection_info:
            info = self.connection_info[websocket]
            teams = info.get("teams", [])
            
            # Remove from each team
            for team in teams:
                if team in self.team_connections:
                    self.team_connections[team].discard(websocket)
                    # Clean up empty team sets
                    if not self.team_connections[team]:
                        del self.team_connections[team]
                        
            # Remove from global
            self.global_connections.discard(websocket)
            
            # Remove connection info
            del self.connection_info[websocket]
            
    async def broadcast_to_team(self, team: str, notification: DocumentNotification):
        """
        Send a notification to all connections subscribed to a specific team.
        
        Args:
            team: The team name to broadcast to
            notification: The notification payload
        """
        message = notification.to_json()
        
        # Get team connections
        connections = self.team_connections.get(team, set()).copy()
        
        # Also include global connections
        connections.update(self.global_connections)
        
        # Send to all connections, handling disconnects
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
                
        # Clean up disconnected
        for ws in disconnected:
            await self.disconnect(ws)
            
    async def broadcast_to_teams(self, teams: list[str], notification: DocumentNotification):
        """
        Send a notification to all connections subscribed to any of the specified teams.
        
        Args:
            teams: List of team names to broadcast to
            notification: The notification payload
        """
        message = notification.to_json()
        
        # Collect unique connections across all teams
        connections: Set[WebSocket] = set()
        for team in teams:
            connections.update(self.team_connections.get(team, set()))
            
        # Also include global connections
        connections.update(self.global_connections)
        
        # Send to all connections
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
                
        # Clean up disconnected
        for ws in disconnected:
            await self.disconnect(ws)
            
    async def broadcast_global(self, notification: DocumentNotification):
        """
        Send a notification to all global/admin connections.
        """
        message = notification.to_json()
        
        disconnected = []
        for websocket in self.global_connections.copy():
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
                
        for ws in disconnected:
            await self.disconnect(ws)
            
    def get_connection_stats(self) -> dict:
        """
        Get statistics about current connections.
        """
        return {
            "total_connections": len(self.connection_info),
            "global_connections": len(self.global_connections),
            "teams": {
                team: len(connections) 
                for team, connections in self.team_connections.items()
            }
        }


# Global connection manager instance
manager = ConnectionManager()


# Helper functions for creating notifications

def create_document_created_notification(
    document_id: str,
    team: str,
    title: str,
    categories: list[str]
) -> DocumentNotification:
    """Create a notification for when a new document is created/classified"""
    return DocumentNotification(
        event_type="document_created",
        document_id=document_id,
        team=team,
        title=title,
        categories=categories,
        timestamp=datetime.utcnow().isoformat()
    )


def create_action_performed_notification(
    document_id: str,
    team: str,
    action_type: str,
    performed_by: str
) -> DocumentNotification:
    """Create a notification for when an action is performed on a document"""
    return DocumentNotification(
        event_type="action_performed",
        document_id=document_id,
        team=team,
        action_type=action_type,
        performed_by=performed_by,
        timestamp=datetime.utcnow().isoformat()
    )


def create_status_changed_notification(
    document_id: str,
    team: str,
    new_status: str,
    performed_by: str
) -> DocumentNotification:
    """Create a notification for when document status changes"""
    return DocumentNotification(
        event_type="status_changed",
        document_id=document_id,
        team=team,
        new_status=new_status,
        performed_by=performed_by,
        timestamp=datetime.utcnow().isoformat()
    )


def create_priority_changed_notification(
    document_id: str,
    team: str,
    priority: str,
    performed_by: str
) -> DocumentNotification:
    """Create a notification for when document priority changes"""
    return DocumentNotification(
        event_type="priority_changed",
        document_id=document_id,
        team=team,
        priority=priority,
        performed_by=performed_by,
        timestamp=datetime.utcnow().isoformat()
    )
