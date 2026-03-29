import numpy as np
import heapq
import math
from typing import Dict, List, Tuple, Optional
from src.models.node import LunarNode
from src.core.config import settings

class LunarSurfaceGraph:
    """Models the lunar surface as a grid-based navigational graph."""
    
    def __init__(self):
        self.grid_size = settings.GRID_SIZE
        self.nodes: Dict[str, LunarNode] = {}
        self.sun_azimuth = 45.0
        self.sun_elevation = 15.0
        
        # Benchmarking metrics
        self.last_visited_nodes = 0
    
    @staticmethod
    def get_node_id(x: int, y: int) -> str:
        return f"{x}_{y}"
    
    def build_graph(self, heightmap: np.ndarray, roughness_map: np.ndarray, crater_rim_map: np.ndarray):
        """Constructs the nodes and edges from terrain maps."""
        n = self.grid_size
        
        # 1. Create nodes
        for y in range(n):
            for x in range(n):
                idx = self.get_node_id(x, y)
                node = LunarNode(
                    id_str=idx, grid_x=x, grid_y=y,
                    elevation=heightmap[y, x],
                    roughness=roughness_map[y, x]
                )
                node.is_crater_rim = crater_rim_map[y, x]
                self.nodes[idx] = node
        
        # 2. Calculate slopes
        self._calculate_slopes(heightmap)
        
        # 3. Calculate initial illumination
        self._calculate_illumination(heightmap)
        
        # 4. Connect 8-way neighbors
        self._connect_neighbors()
    
    def _calculate_slopes(self, heightmap: np.ndarray):
        """Gradient calculation using neighboring differences (Sobel-like)."""
        n = self.grid_size
        cs = settings.CELL_SIZE_METERS
        for y in range(n):
            for x in range(n):
                dh_dx = 0.0
                dh_dy = 0.0
                
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        diff = (heightmap[ny, nx] - heightmap[y, x]) / cs
                        if dx != 0: dh_dx += diff
                        if dy != 0: dh_dy += diff
                
                gradient = np.sqrt(dh_dx**2 + dh_dy**2)
                slope_deg = np.degrees(np.arctan(gradient))
                self.nodes[self.get_node_id(x, y)].slope = slope_deg
    
    def _calculate_illumination(self, heightmap: np.ndarray):
        """Raycast-like shadow checking based on sun angle."""
        n = self.grid_size
        azimuth_rad = np.radians(self.sun_azimuth)
        
        sun_dx = np.cos(azimuth_rad)
        sun_dy = np.sin(azimuth_rad)
        
        for y in range(n):
            for x in range(n):
                node = self.nodes[self.get_node_id(x, y)]
                shadow_factor = 1.0
                check_dist = 5
                
                for d in range(1, check_dist + 1):
                    cx = int(x + sun_dx * d)
                    cy = int(y + sun_dy * d)
                    
                    if 0 <= cx < n and 0 <= cy < n:
                        h_diff = heightmap[cy, cx] - heightmap[y, x]
                        distance = d * settings.CELL_SIZE_METERS
                        blocking_angle = np.degrees(np.arctan2(h_diff, distance))
                        
                        if blocking_angle > self.sun_elevation:
                            shadow_factor *= 0.2
                            break
                
                node.illumination = max(0.05, shadow_factor)

    def _connect_neighbors(self):
        """Connect all nodes to adjacent 8 cells and calculate costs."""
        n = self.grid_size
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for y in range(n):
            for x in range(n):
                node = self.nodes[self.get_node_id(x, y)]
                node.neighbors = [] # Clear prior
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < n:
                        neighbor = self.nodes[self.get_node_id(nx, ny)]
                        cost = self.calculate_edge_cost(node, neighbor)
                        node.add_neighbor(neighbor, cost)

    def calculate_edge_cost(self, u: LunarNode, v: LunarNode) -> float:
        """
        Multifactor cost function:
        Cost = 3D_Distance * Slope_Factor * Energy_Factor * Hazard_Factor * Crater_Penalty
        """
        cs = settings.CELL_SIZE_METERS
        dx = (v.grid_x - u.grid_x) * cs
        dy = (v.grid_y - u.grid_y) * cs
        dh = v.elevation - u.elevation
        distance_3d = np.sqrt(dx**2 + dy**2 + dh**2)
        
        # Slope factor
        avg_slope = (u.slope + v.slope) / 2.0
        slope_factor = 1.0 + settings.SLOPE_PENALTY * avg_slope
        
        # Energy factor
        shadow_penalty = settings.SHADOW_PENALTY * (1.0 - v.illumination)
        rough_penalty = settings.ROUGHNESS_PENALTY * v.roughness
        energy_factor = 1.0 + shadow_penalty + rough_penalty
        
        # Hazard factor (PyCBC)
        hazard_factor = 1.0 + settings.HAZARD_PENALTY * v.hazard_score
        
        # Crater Zone factor (YOLO)
        crater_factor = 1.0
        if v.is_crater_no_go:
            crater_factor = settings.CRATER_ZONE_PENALTY
        elif v.is_crater_rim:
            crater_factor = settings.CRATER_RIM_PENALTY
            
        return distance_3d * slope_factor * energy_factor * hazard_factor * crater_factor

    def apply_crater_map(self, crater_map: np.ndarray):
        n = self.grid_size
        for y in range(n):
            for x in range(n):
                self.nodes[self.get_node_id(x, y)].is_crater_no_go = bool(crater_map[y, x])

    def apply_hazard_map(self, hazard_map: np.ndarray):
        n = self.grid_size
        for y in range(n):
            for x in range(n):
                self.nodes[self.get_node_id(x, y)].hazard_score = hazard_map[y, x]

    def update_costs(self, time_offset_hours: float = 0.0):
        """Update sun position and recalculate graph edges."""
        self.sun_azimuth = (45.0 + time_offset_hours * 7.5) % 360
        self.sun_elevation = max(2.0, 15.0 + 5.0 * np.sin(np.radians(time_offset_hours * 15)))
        
        # Quick sync heightmap for shadow calc
        n = self.grid_size
        heightmap = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                heightmap[y, x] = self.nodes[self.get_node_id(x, y)].elevation
                
        self._calculate_illumination(heightmap)
        self._connect_neighbors()

    def dijkstra_search(self, start_id: str, target_id: str) -> Optional[Tuple[List[LunarNode], float]]:
        self.last_visited_nodes = 0
        distances = {start_id: 0.0}
        previous = {start_id: None}
        heap = [(0.0, start_id)]
        
        while heap:
            self.last_visited_nodes += 1
            current_distance, u_id = heapq.heappop(heap)
            
            if current_distance > distances.get(u_id, float('inf')):
                continue
                
            if u_id == target_id:
                path = []
                curr = target_id
                while curr is not None:
                    path.append(self.nodes[curr])
                    curr = previous.get(curr)
                path.reverse()
                return path, distances[target_id]
                
            for neighbor, cost in self.nodes[u_id].neighbors:
                new_distance = current_distance + cost
                if new_distance < distances.get(neighbor.id, float('inf')):
                    distances[neighbor.id] = new_distance
                    previous[neighbor.id] = u_id
                    heapq.heappush(heap, (new_distance, neighbor.id))
                    
        return None
