from fastapi import APIRouter
from src.api.v1.endpoints import routes, chat

api_router = APIRouter()
api_router.include_router(routes.router, prefix="/routes", tags=["Routes"])
api_router.include_router(chat.router, prefix="/chat", tags=["AI Assistant"])
