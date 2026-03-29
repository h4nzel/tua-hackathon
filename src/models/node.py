from typing import List, Tuple

class LunarNode:
    """Represents a grid point on the lunar surface."""
    
    # Using slots for memory efficiency in A* with 10k+ nodes
    __slots__ = [
        'id', 'grid_x', 'grid_y', 'elevation', 'roughness', 
        'slope', 'illumination', 'is_crater_rim', 'hazard_score', 
        'is_crater_no_go', 'neighbors'
    ]
    
    def __init__(self, id_str: str, grid_x: int, grid_y: int, 
                 elevation: float = 0.0, roughness: float = 0.0):
        self.id = id_str
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.elevation = elevation
        self.roughness = roughness
        self.slope = 0.0
        self.illumination = 1.0
        self.is_crater_rim = False
        self.hazard_score = 0.0
        self.is_crater_no_go = False
        self.neighbors: List[Tuple['LunarNode', float]] = []
    
    def add_neighbor(self, node: 'LunarNode', cost: float):
        self.neighbors.append((node, cost))
    
    def __repr__(self):
        return f"LunarNode({self.grid_x},{self.grid_y} h={self.elevation:.1f}m)"
