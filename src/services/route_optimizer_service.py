import numpy as np
import heapq
import time
import torch
import torch.optim as optim
import logging
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data
from src.core.graph import LunarSurfaceGraph
from src.core.config import settings
from src.models.node import LunarNode
from src.infrastructure.ml.gnn_landmark import GNNLandmarkSelector

logger = logging.getLogger("LunarRouter.Optimizer")

class RouteOptimizerService:
    """Service wrapping DL-ALT (Deep Learning A* Landmark Triangle Inequality)."""
    
    def __init__(self, lunar_graph: LunarSurfaceGraph):
        self.graph = lunar_graph
        self.gnn_model: Optional[GNNLandmarkSelector] = None
        self.landmarks: List[LunarNode] = []
        self.dist_from_landmarks: Dict[str, Dict[str, float]] = {}
        self.last_visited_nodes = 0
        
    def _prepare_gnn_data(self) -> Tuple[Data, dict]:
        """Prepares PyTorch Geometric Data object from the grid nodes."""
        node_features = []
        node_mapping = {}
        idx_mapping = {}
        
        # Mapping grid coordinates safely
        for i, (idx, node) in enumerate(self.graph.nodes.items()):
            node_mapping[idx] = i
            idx_mapping[i] = idx
            
        edge_index = [[], []]
        for idx, node in self.graph.nodes.items():
            u = node_mapping[idx]
            
            # Normalize elevation exactly between 0 and 1
            elev_norm = (node.elevation - settings.MIN_ELEVATION) / (settings.MAX_ELEVATION - settings.MIN_ELEVATION)
            elev_norm = max(0.0, min(1.0, elev_norm))
            
            feat = [
                elev_norm, 
                node.slope / 90.0,
                node.illumination,
                node.roughness,
                1.0 if node.is_crater_rim else 0.0,
                len(node.neighbors) / 8.0
            ]
            node_features.append(feat)
            
            for nb, _ in node.neighbors:
                v = node_mapping[nb.id]
                edge_index[0].append(u)
                edge_index[1].append(v)
                
        x = torch.tensor(node_features, dtype=torch.float)
        edges = torch.tensor(edge_index, dtype=torch.long)
        
        data = Data(x=x, edge_index=edges)
        return data, idx_mapping

    def _train_landmark_selector(self, epochs: int = 30):
        if not self.gnn_model:
            self.gnn_model = GNNLandmarkSelector(in_channels=6, hidden_channels=16, out_channels=1)
            
        data, _ = self._prepare_gnn_data()
        optimizer = optim.Adam(self.gnn_model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        target = data.x[:, 0:1] * 0.5 + data.x[:, 4:5] * 0.5 
        
        self.gnn_model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            out = self.gnn_model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        
        logger.info(f"GNN Landmark Selector trained successfully. MSE Loss: {loss.item():.4f}")

    def _select_landmarks(self, k: int = 8):
        """Strategical placement of routing landmarks based on GNN output."""
        if not self.gnn_model:
            self._train_landmark_selector()
            
        data, node_mapping = self._prepare_gnn_data()
        self.gnn_model.eval()
        
        with torch.no_grad():
            importance = self.gnn_model(data).cpu().numpy().flatten()
            
        sorted_indices = np.argsort(importance)[::-1]
        nodes_list = list(self.graph.nodes.values())
        
        min_distance = settings.GRID_SIZE // (k // 2)
        selected = []
        
        for idx in sorted_indices:
            node = nodes_list[idx]
            
            too_close = False
            for existing in selected:
                dist = np.sqrt((node.grid_x - existing.grid_x)**2 + (node.grid_y - existing.grid_y)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(node)
                if len(selected) == k:
                    break
                    
        self.landmarks = selected
        logger.info(f"Selected {len(self.landmarks)} landmarks for efficient routing.")
        
        # Precompute dijkstra grids for landmarks
        for l_node in self.landmarks:
            _, dists = self._run_dijkstra_full(l_node.id)
            self.dist_from_landmarks[l_node.id] = dists

    def _run_dijkstra_full(self, start_id: str) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
        distances = {start_id: 0.0}
        previous = {start_id: None}
        heap = [(0.0, start_id)]
        
        while heap:
            curr_dist, u_id = heapq.heappop(heap)
            if curr_dist > distances.get(u_id, float('inf')):
                continue
                
            for v, cost in self.graph.nodes[u_id].neighbors:
                new_dist = curr_dist + cost
                if new_dist < distances.get(v.id, float('inf')):
                    distances[v.id] = new_dist
                    previous[v.id] = u_id
                    heapq.heappush(heap, (new_dist, v.id))
                    
        return previous, distances

    def _dl_alt_heuristic(self, u: LunarNode, t: LunarNode) -> float:
        """Triangle inequality heuristic using precalculated distances."""
        max_h = 0.0
        for landmark in self.landmarks:
            dist_lu = self.dist_from_landmarks[landmark.id].get(u.id, float('inf'))
            dist_lv = self.dist_from_landmarks[landmark.id].get(t.id, float('inf'))
            if dist_lu != float('inf') and dist_lv != float('inf'):
                max_h = max(max_h, abs(dist_lu - dist_lv))
        return max_h

    def calculate_route_dlalt(self, 
                             start_id: str, 
                             target_id: str, 
                             time_offset: float = 0.0) -> Optional[Tuple[List[LunarNode], float, int]]:
        """A* optimized path finding using DL-ALT. Returns (route, cost, nodes_explored)."""
        logger.info(f"Finding DL-ALT route from {start_id} to {target_id}...")
        
        if time_offset > 0:
            self.graph.update_costs(time_offset)
            
        if not self.landmarks:
            self._select_landmarks(k=5)
            
        start = self.graph.nodes.get(start_id)
        target = self.graph.nodes.get(target_id)
        if not start or not target:
            return None
            
        h_start = self._dl_alt_heuristic(start, target)
        open_set = [(h_start, 0.0, id(start), start_id)]
        g_scores = {start_id: 0.0}
        previous = {start_id: None}
        
        self.last_visited_nodes = 0
        
        while open_set:
            f_score, g, _, curr_id = heapq.heappop(open_set)
            self.last_visited_nodes += 1
            
            if curr_id == target_id:
                route = []
                trace = target_id
                while trace:
                    route.append(self.graph.nodes[trace])
                    trace = previous.get(trace)
                route.reverse()
                return route, g, self.last_visited_nodes
                
            if g > g_scores.get(curr_id, float('inf')):
                continue
                
            curr_node = self.graph.nodes[curr_id]
            for nb, cost in curr_node.neighbors:
                new_g = g + cost
                if new_g < g_scores.get(nb.id, float('inf')):
                    g_scores[nb.id] = new_g
                    previous[nb.id] = curr_id
                    f = new_g + self._dl_alt_heuristic(nb, target)
                    heapq.heappush(open_set, (f, new_g, id(nb), nb.id))
                    
        return None
