from fastapi import FastAPI
from routers import users, documents
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

app = FastAPI(
    title="My Awesome API",
    description="API with document processing capabilities using LangChain and LlamaIndex",
    version="1.0.0"
)

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    APP_NAME: str = "DefaultApp"  # Default value if not found in .env

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
app = FastAPI(title=settings.APP_NAME)


# Include routers from other modules
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])

@app.get("/")
def root():
    return {"message": "API is running"}
