import time
import logging
from typing import Tuple
from fastapi import HTTPException
from src.core.graph import LunarSurfaceGraph
from src.core.config import settings
from src.models.schemas import RouteRequest, RouteResponse, RouteSummary
from src.services.terrain_service import TerrainGeneratorService
from src.services.crater_detection_service import CraterDetectionService
from src.services.hazard_detection_service import HazardDetectionService
from src.services.route_optimizer_service import RouteOptimizerService
from src.utils.visualizer import LunarVisualizer

logger = logging.getLogger("LunarRouter.Controller")

class RouteController:
    """Coordinates HTTP traffic and business logic for route planning."""
    
    def __init__(
        self,
        terrain_service: TerrainGeneratorService,
        crater_service: CraterDetectionService,
        hazard_service: HazardDetectionService
    ):
        self.terrain_service = terrain_service
        self.crater_service = crater_service
        self.hazard_service = hazard_service
        
        # Singleton graph representation inside controller for memory efficiency
        self.graph = LunarSurfaceGraph()
        self.optimizer = RouteOptimizerService(self.graph)
        self._is_graph_ready = False

    def build_world(self):
        """Heavy initialization mapping satellite data to AI models."""
        if self._is_graph_ready:
            return
            
        logger.info("Building Lunar World Map...")
        h_map, r_map, c_rim, source_info = self.terrain_service.get_terrain_data()
        logger.info(f"Loaded terrain source: {source_info}")
        
        c_map = self.crater_service.detect_craters(settings.YOLO_TEST_IMAGE_PATH)
        hz_map = self.hazard_service.create_hazard_map(r_map, c_rim)
        
        self.graph.build_graph(h_map, r_map, c_rim)
        self.graph.apply_crater_map(c_map)
        self.graph.apply_hazard_map(hz_map)
        self.graph.update_costs(0.0)
        
        self._is_graph_ready = True
        logger.info("World state is ready.")

    def calculate_route(self, request: RouteRequest) -> RouteResponse:
        self.build_world() # Ensure lazy load is complete
        
        start_id = self.graph.get_node_id(request.start.x, request.start.y)
        target_id = self.graph.get_node_id(request.target.x, request.target.y)
        
        if start_id not in self.graph.nodes or target_id not in self.graph.nodes:
            raise HTTPException(status_code=400, detail="Coordinates are out of grid bounds.")
            
        t0 = time.perf_counter()
        
        if request.algorithm.lower() == 'dl-alt':
            res = self.optimizer.calculate_route_dlalt(start_id, target_id, request.time_offset_hours)
            if not res:
                raise HTTPException(status_code=404, detail="No passable route found.")
            route_path, cost, nodes_explored = res
        else:
            res = self.graph.dijkstra_search(start_id, target_id)
            if not res:
                raise HTTPException(status_code=404, detail="No passable route found.")
            route_path, cost = res
            nodes_explored = self.graph.last_visited_nodes
            
        t_ms = (time.perf_counter() - t0) * 1000
        
        # Render visualizer in background or sync
        try:
            LunarVisualizer.render(
                graph=self.graph,
                route_dlalt=route_path if request.algorithm.lower() == 'dl-alt' else None,
                route_dijkstra=route_path if request.algorithm.lower() != 'dl-alt' else None
            )
        except Exception as e:
            logger.error(f"Failed to render viz: {e}")
            
        coords = [(n.grid_x, n.grid_y) for n in route_path]
        
        return RouteResponse(
            success=True,
            message=f"Optimal route mapped successfully via {request.algorithm}",
            route=RouteSummary(
                path_length=len(route_path),
                cost=cost,
                computation_time_ms=t_ms,
                nodes_explored=nodes_explored,
                path_coordinates=coords
            )
        )
