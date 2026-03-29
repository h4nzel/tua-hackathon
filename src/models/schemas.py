from pydantic import BaseModel, Field
from typing import List, Tuple, Optional

class Point2D(BaseModel):
    x: int = Field(..., ge=0, description="X coordinate on grid")
    y: int = Field(..., ge=0, description="Y coordinate on grid")

class RouteRequest(BaseModel):
    start: Point2D
    target: Point2D
    time_offset_hours: float = Field(0.0, description="Time shift in hours for shadow calculation")
    algorithm: str = Field("dl-alt", description="dl-alt or dijkstra")

class RouteSummary(BaseModel):
    path_length: int
    cost: float
    computation_time_ms: float
    nodes_explored: int
    path_coordinates: List[Tuple[int, int]]

class RouteResponse(BaseModel):
    success: bool
    message: str
    route: Optional[RouteSummary] = None

class ChatMessageResponse(BaseModel):
    reply: str
    route_details: Optional[RouteSummary] = None
    visualization_url: Optional[str] = None
