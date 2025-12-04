from fastapi import FastAPI
from routers import users

app = FastAPI(title="My Awesome API")

# Include routers from other modules
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])

@app.get("/")
def root():
    return {"message": "API is running"}
