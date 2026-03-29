import numpy as np
import scipy.ndimage
from typing import Tuple
from src.infrastructure.data.heightmap_loader import HeightmapLoader
from src.core.config import settings

class TerrainGeneratorService:
    """Service handling synthetic or real map injection to the graph coordinate system."""
    
    def __init__(self, seed: int = 42):
        self.grid_size = settings.GRID_SIZE
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
    def generate_synthetic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Provides fallback Synthetic terrain (Perlin Noise approximation)."""
        n = self.grid_size
        base = self.rng.uniform(0, 1000, (n//10, n//10))
        heightmap = scipy.ndimage.zoom(base, 10, order=3)
        
        # Add craters
        crater_rim_map = np.zeros((n, n), dtype=bool)
        for _ in range(50):
            cx, cy = self.rng.integers(0, n, size=2)
            r = self.rng.integers(5, 15)
            
            y, x = np.ogrid[-cy:n-cy, -cx:n-cx]
            mask = x*x + y*y <= r*r
            rim_mask = (x*x + y*y >= (r-2)**2) & mask
            
            heightmap[mask] -= 300
            heightmap[rim_mask] += 150
            crater_rim_map[rim_mask] = True
            
        # Add high freq noise
        noise = self.rng.normal(0, 50, (n, n))
        heightmap += scipy.ndimage.gaussian_filter(noise, sigma=1)
        
        roughness_map = np.clip(np.abs(noise)/200.0, 0, 1)
        return heightmap, roughness_map, crater_rim_map

    def get_terrain_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Loads from file if possible, otherwise falls back to synthetic generation."""
        import os
        if os.path.exists(settings.HEIGHTMAP_PATH):
            loader = HeightmapLoader(
                settings.HEIGHTMAP_PATH,
                self.grid_size,
                settings.MIN_ELEVATION,
                settings.MAX_ELEVATION
            )
            h, r, c = loader.load()
            return h, r, c, "Satellite Image (LRO)"
        else:
            h, r, c = self.generate_synthetic()
            return h, r, c, "Synthetic Procedural Engine"
