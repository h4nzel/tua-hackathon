import os
import logging
from typing import Optional, Tuple
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import content_types

from src.core.config import settings
from src.controllers.route_controller import RouteController
from src.models.schemas import RouteRequest, Point2D, RouteSummary

logger = logging.getLogger("LunarRouter.AIAgent")

class AgentExecutionResult(BaseModel):
    reply_text: str
    route_summary: Optional[RouteSummary] = None

class AIAgentService:
    """Manages conversational state and tool execution via Gemini API."""
    
    def __init__(self, route_controller: RouteController):
        self.controller = route_controller
        self.api_key = settings.GEMINI_API_KEY
        self.is_configured = False
        self.model = None
        self.last_route_summary: Optional[RouteSummary] = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.is_configured = True
            
            # Initialize model with function calling tools
            self.model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                tools=[self.calculate_lunar_route],
                system_instruction=(
                    "You are a professional Lunar Navigation AI Assistant for a lunar rover mission. "
                    "You have direct access to a sophisticated route calculation system using DL-ALT, "
                    "PyCBC (Doppler Hazard mapping), and YOLO (Crater Detection). "
                    "When the user asks you to find a route between two coordinates, you MUST use the "
                    "`calculate_lunar_route` tool. Always be concise, technical, and professional."
                )
            )
        else:
            logger.warning("GEMINI_API_KEY not set. Chat Assistant will be disabled.")

    def calculate_lunar_route(self, start_x: int, start_y: int, target_x: int, target_y: int, algorithm: str = "dl-alt") -> str:
        """
        Calculates a safe optimal route avoiding craters and hazards.
        Args:
            start_x: The starting X coordinate (0-99).
            start_y: The starting Y coordinate (0-99).
            target_x: The target X coordinate (0-99).
            target_y: The target Y coordinate (0-99).
            algorithm: The algorithm to use. Default is "dl-alt", can also be "dijkstra".
        """
        try:
            req = RouteRequest(
                start=Point2D(x=start_x, y=start_y),
                target=Point2D(x=target_x, y=target_y),
                algorithm=algorithm,
                time_offset_hours=0.0
            )
            logger.info(f"AI requested route calculation: {req}")
            
            response = self.controller.calculate_route(req)
            self.last_route_summary = response.route
            
            return (
                f"SUCCESS: Route calculated! "
                f"Path length: {response.route.path_length} steps. "
                f"Nodes explored: {response.route.nodes_explored}. "
                f"Computation time: {response.route.computation_time_ms:.1f}ms. "
                f"Visual map generated at lunar_route_result.png."
            )
        except Exception as e:
            return f"FAILED to calculate route: {str(e)}"

    def chat_with_agent(self, prompt: str, image_bytes: Optional[bytes] = None, mime_type: str = "image/jpeg") -> AgentExecutionResult:
        if not self.is_configured:
            return AgentExecutionResult(reply_text="AI Assistant is disabled. Please configure GEMINI_API_KEY in the environment or .env file.")
            
        self.last_route_summary = None
        
        contents = []
        if image_bytes:
            contents.append({
                "mime_type": mime_type,
                "data": image_bytes
            })
        contents.append(prompt)

        try:
            chat = self.model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(contents)
            
            return AgentExecutionResult(
                reply_text=response.text,
                route_summary=self.last_route_summary
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return AgentExecutionResult(reply_text=f"An error occurred communicating with my AI core: {e}")
