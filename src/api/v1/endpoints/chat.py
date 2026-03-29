from fastapi import APIRouter, Depends, Form, File, UploadFile
from typing import Optional
import logging
from src.controllers.route_controller import RouteController
from src.services.terrain_service import TerrainGeneratorService
from src.services.crater_detection_service import CraterDetectionService
from src.services.hazard_detection_service import HazardDetectionService
from src.services.ai_agent_service import AIAgentService
from src.models.schemas import ChatMessageResponse

router = APIRouter()
logger = logging.getLogger("LunarRouter.API.Chat")

# Dependency injection
def get_ai_agent() -> AIAgentService:
    terrain_service = TerrainGeneratorService()
    crater_service = CraterDetectionService()
    hazard_service = HazardDetectionService()
    controller = RouteController(terrain_service, crater_service, hazard_service)
    return AIAgentService(controller)

@router.post("/message", response_model=ChatMessageResponse)
async def send_chat_message(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    agent: AIAgentService = Depends(get_ai_agent)
):
    """
    Interact with the Lunar Rover Gemini Assistant.
    Send natural language instructions (e.g., 'Chart a route from 10,10 to 90,90').
    Supports optional image uploads for computer vision context.
    """
    logger.info(f"Chat request received. Image included: {image is not None}")
    
    image_bytes = None
    mime_type = "image/jpeg"
    if image:
        image_bytes = await image.read()
        mime_type = image.content_type

    result = agent.chat_with_agent(prompt, image_bytes, mime_type)
    
    # Send response back
    response = ChatMessageResponse(
        reply=result.reply_text,
        route_details=result.route_summary,
        visualization_url="/lunar_route_result.png" if result.route_summary else None
    )
    
    return response
