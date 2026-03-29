from fastapi import APIRouter, Depends
from src.controllers.route_controller import RouteController
from src.services.terrain_service import TerrainGeneratorService
from src.services.crater_detection_service import CraterDetectionService
from src.services.hazard_detection_service import HazardDetectionService
from src.models.schemas import RouteRequest, RouteResponse
import logging

router = APIRouter()
logger = logging.getLogger("LunarRouter.API")

# Dependency injection
def get_route_controller() -> RouteController:
    # In a real app, these would be cached/singletons via lifespan events
    terrain_service = TerrainGeneratorService()
    crater_service = CraterDetectionService()
    hazard_service = HazardDetectionService()
    return RouteController(terrain_service, crater_service, hazard_service)

@router.post("/calculate", response_model=RouteResponse)
def calculate_optimal_route(
    request: RouteRequest,
    controller: RouteController = Depends(get_route_controller)
):
    """
    Given a starting point (x,y) and target point (x,y), calculate the optimal route.
    Accounts for slope, roughness, illumination, PyCBC hazards, and YOLO craters.
    Generates a 8-panel matplotlib dashboard in the root folder upon completion.
    """
    logger.info(f"Received route request: {request.start} -> {request.target}")
    return controller.calculate_route(request)
