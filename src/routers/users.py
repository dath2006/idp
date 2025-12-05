"""
User Authentication and Management Router.

Handles:
- User registration and login
- JWT token management
- User profile and team membership
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from services.auth_service import (
    Token,
    UserCreate,
    UserResponse,
    LoginRequest,
    authenticate_user,
    register_user,
    get_current_user,
    user_to_response,
    user_has_team_access,
    VALID_TEAMS,
)
from services.mongodb_service import User, get_mongodb, UserRole


router = APIRouter()
security = HTTPBearer()


# ============================================================
# Dependency: Get Current User
# ============================================================

async def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Dependency to get the current authenticated user."""
    token = credentials.credentials
    user = await get_current_user(token)
    
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def get_admin_user(
    user: User = Depends(get_authenticated_user)
) -> User:
    """Dependency to get an admin user."""
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return user


# ============================================================
# Authentication Endpoints
# ============================================================

@router.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """
    Register a new user.
    
    Teams must be from: compliance, engineer, finance, hr, legal,
    operations, procurement, project_leader, safety
    """
    # Validate teams
    for team in user_data.teams:
        if team.lower() not in VALID_TEAMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid team: {team}. Valid teams: {VALID_TEAMS}"
            )
    
    # Normalize team names to lowercase
    user_data.teams = [t.lower() for t in user_data.teams]
    
    user = await register_user(user_data)
    if user is None:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    return user_to_response(user)


@router.post("/auth/login", response_model=Token)
async def login(login_data: LoginRequest):
    """
    Login and get JWT token.
    """
    from services.auth_service import create_access_token
    
    user = await authenticate_user(login_data.email, login_data.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    return create_access_token(user)


@router.get("/auth/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_authenticated_user)):
    """
    Get current user profile.
    """
    return user_to_response(user)


@router.put("/auth/me", response_model=UserResponse)
async def update_me(
    name: Optional[str] = None,
    user: User = Depends(get_authenticated_user)
):
    """
    Update current user profile.
    """
    mongodb = await get_mongodb()
    
    updates = {}
    if name:
        updates["name"] = name
    
    if updates:
        await mongodb.update_document(user.id, updates)
        user.name = name or user.name
    
    return user_to_response(user)


# ============================================================
# Team Endpoints
# ============================================================

class TeamInfo(BaseModel):
    """Team information."""
    id: str
    name: str
    display_name: str
    member_count: int = 0


@router.get("/teams", response_model=List[TeamInfo])
async def get_teams(user: User = Depends(get_authenticated_user)):
    """
    Get all teams (with member counts).
    """
    mongodb = await get_mongodb()
    
    team_info = []
    for team_id in VALID_TEAMS:
        users = await mongodb.get_users_by_team(team_id)
        team_info.append(TeamInfo(
            id=team_id,
            name=team_id,
            display_name=team_id.replace("_", " ").title(),
            member_count=len(users)
        ))
    
    return team_info


@router.get("/teams/{team_id}/members", response_model=List[UserResponse])
async def get_team_members(
    team_id: str,
    user: User = Depends(get_authenticated_user)
):
    """
    Get members of a team.
    """
    if team_id.lower() not in VALID_TEAMS:
        raise HTTPException(status_code=404, detail="Team not found")
    
    if not user_has_team_access(user, team_id.lower()):
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    mongodb = await get_mongodb()
    users = await mongodb.get_users_by_team(team_id.lower())
    
    return [user_to_response(u) for u in users]


# ============================================================
# Admin User Management
# ============================================================

@router.get("/admin/users", response_model=List[UserResponse])
async def get_all_users(admin: User = Depends(get_admin_user)):
    """
    Get all users (admin only).
    """
    mongodb = await get_mongodb()
    users = await mongodb.get_all_users()
    return [user_to_response(u) for u in users]


class UserUpdate(BaseModel):
    """User update request."""
    name: Optional[str] = None
    teams: Optional[List[str]] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


@router.put("/admin/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    update_data: UserUpdate,
    admin: User = Depends(get_admin_user)
):
    """
    Update a user (admin only).
    """
    mongodb = await get_mongodb()
    
    target_user = await mongodb.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    updates = {}
    
    if update_data.name:
        updates["name"] = update_data.name
    
    if update_data.teams is not None:
        # Validate teams
        for team in update_data.teams:
            if team.lower() not in VALID_TEAMS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid team: {team}"
                )
        updates["teams"] = [t.lower() for t in update_data.teams]
    
    if update_data.role is not None:
        updates["role"] = update_data.role.value
    
    if update_data.is_active is not None:
        updates["is_active"] = update_data.is_active
    
    if updates:
        await mongodb.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
    
    # Fetch updated user
    updated_user = await mongodb.get_user_by_id(user_id)
    return user_to_response(updated_user)


# Import ObjectId at the top
from bson import ObjectId
