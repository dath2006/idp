"""
Authentication Service for JWT-based authentication.

Handles:
- Password hashing and verification
- JWT token generation and validation
- User authentication
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

from services.mongodb_service import User, get_mongodb, UserRole


# ============================================================
# Configuration
# ============================================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================
# Token Models
# ============================================================

class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    user_id: str
    email: str
    teams: list[str]
    role: str


class UserCreate(BaseModel):
    """User creation request."""
    email: str
    password: str
    name: str
    teams: list[str] = []
    role: UserRole = UserRole.MEMBER


class UserResponse(BaseModel):
    """User response (without password)."""
    id: str
    email: str
    name: str
    teams: list[str]
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class LoginRequest(BaseModel):
    """Login request."""
    email: str
    password: str


# ============================================================
# Password Functions
# ============================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================
# JWT Functions
# ============================================================

def create_access_token(
    user: User,
    expires_delta: Optional[timedelta] = None
) -> Token:
    """Create a JWT access token for a user."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire = datetime.utcnow() + expires_delta
    
    to_encode = {
        "sub": user.id,
        "email": user.email,
        "teams": user.teams,
        "role": user.role.value if isinstance(user.role, UserRole) else user.role,
        "exp": expire
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return Token(
        access_token=encoded_jwt,
        expires_in=int(expires_delta.total_seconds())
    )


def decode_access_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        teams: list = payload.get("teams", [])
        role: str = payload.get("role", "member")
        
        if user_id is None or email is None:
            return None
        
        return TokenData(
            user_id=user_id,
            email=email,
            teams=teams,
            role=role
        )
    except JWTError:
        return None


# ============================================================
# Authentication Functions
# ============================================================

async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password."""
    mongodb = await get_mongodb()
    user = await mongodb.get_user_by_email(email)
    
    if not user:
        return None
    
    if not user.is_active:
        return None
    
    if not verify_password(password, user.password_hash):
        return None
    
    # Update last login
    await mongodb.update_user_login(user.id)
    
    return user


async def register_user(user_data: UserCreate) -> Optional[User]:
    """Register a new user."""
    mongodb = await get_mongodb()
    
    # Check if email already exists
    existing = await mongodb.get_user_by_email(user_data.email)
    if existing:
        return None
    
    # Create user with hashed password
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        name=user_data.name,
        teams=user_data.teams,
        role=user_data.role
    )
    
    user_id = await mongodb.create_user(user)
    user.id = user_id
    
    return user


async def get_current_user(token: str) -> Optional[User]:
    """Get the current user from a JWT token."""
    token_data = decode_access_token(token)
    if token_data is None:
        return None
    
    mongodb = await get_mongodb()
    user = await mongodb.get_user_by_id(token_data.user_id)
    
    if user is None or not user.is_active:
        return None
    
    return user


def user_to_response(user: User) -> UserResponse:
    """Convert User to UserResponse (without password)."""
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        teams=user.teams,
        role=user.role.value if isinstance(user.role, UserRole) else user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )


# ============================================================
# Team Access Validation
# ============================================================

def user_has_team_access(user: User, team: str) -> bool:
    """Check if user has access to a team."""
    if user.role == UserRole.ADMIN:
        return True
    return team in user.teams


def user_can_modify(user: User, team: str) -> bool:
    """Check if user can modify documents in a team."""
    if user.role == UserRole.ADMIN:
        return True
    if user.role == UserRole.VIEWER:
        return False
    return team in user.teams


# Valid teams matching the ML model output
VALID_TEAMS = [
    "compliance",
    "engineer", 
    "finance",
    "hr",
    "legal",
    "operations",
    "procurement",
    "project_leader",
    "safety"
]


def validate_team(team: str) -> bool:
    """Validate that a team name is valid."""
    return team.lower() in VALID_TEAMS
