# src/routers/users.py
from urllib import request
from fastapi import APIRouter, HTTPException
from services import user_service

router = APIRouter()

@router.get("/")
async def read_users(msg: str = None):
    # Call service layer logic
    users = await user_service.get_all_users(msg)
    return users

@router.get("/{user_id}")
def read_user(user_id: int):
    # Call service layer logic
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
