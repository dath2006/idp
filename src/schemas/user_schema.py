# src/schemas/user_schemas.py
"""
User Schemas for IDP System.

Includes role-based access control and department assignments
for intelligent document routing and dissemination.
"""

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Department(str, Enum):
    """Available departments (mirrors config.departments)."""
    ENGINEERING = "engineering"
    FINANCE = "finance"
    PROCUREMENT = "procurement"
    HR = "hr"
    OPERATIONS = "operations"
    SAFETY = "safety"
    COMPLIANCE = "compliance"
    LEGAL = "legal"
    MANAGEMENT = "management"


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., description="User's email address")
    role: UserRole = Field(default=UserRole.VIEWER, description="User's role in the system")
    department: Department = Field(..., description="User's primary department")
    secondary_departments: List[Department] = Field(
        default_factory=list, 
        description="Additional departments user has access to"
    )
    notification_enabled: bool = Field(default=True, description="Whether user receives email notifications")


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[str] = None
    role: Optional[UserRole] = None
    department: Optional[Department] = None
    secondary_departments: Optional[List[Department]] = None
    notification_enabled: Optional[bool] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """Schema for user responses."""
    id: int
    name: str
    email: str
    role: UserRole
    department: Department
    secondary_departments: List[Department] = Field(default_factory=list)
    notification_enabled: bool = True
    is_active: bool = True


class UserNotificationPreference(BaseModel):
    """Schema for user notification preferences."""
    user_id: int
    email_enabled: bool = True
    document_types: List[str] = Field(
        default_factory=list,
        description="Document types the user wants notifications for (empty = all)"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that trigger notifications for this user"
    )

