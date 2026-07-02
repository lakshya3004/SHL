from fastapi import APIRouter
from app.api.routes import health, chat

api_router = APIRouter()

# API versioned routes (for documentation and backwards compat)
api_router.include_router(health.router, prefix="/health", tags=["system"])
api_router.include_router(chat.router, prefix="/chat", tags=["recommendation"])
