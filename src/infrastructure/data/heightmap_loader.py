import numpy as np
from typing import Tuple
from PIL import Image
import warnings
from scipy.ndimage import zoom, sobel, uniform_filter

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class HeightmapLoader:
    """Loads and scales raster heightmaps into grids with geomorphology features."""
    
    def __init__(self, file_path: str, grid_size: int, alt_min: float, alt_max: float):
        self.file_path = file_path
        self.grid_size = grid_size
        self.alt_min = alt_min
        self.alt_max = alt_max

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns heightmap, roughness_map, crater_rim_map."""
        try:
            img = Image.open(self.file_path).convert('L')
            img_array = np.array(img, dtype=np.float64)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load heightmap: {self.file_path} -> {e}")

        # Scale to grid size via bilinear interpolation
        scale_y = self.grid_size / img_array.shape[0]
        scale_x = self.grid_size / img_array.shape[1]
        img_scaled = zoom(img_array, (scale_y, scale_x), order=1)
        
        # 0-255 -> Altitude conversion
        heightmap = self.alt_min + (img_scaled / 255.0) * (self.alt_max - self.alt_min)
        
        # Crater Rim Detection (Sobel Filter + 85th percentile thresholding)
        grad_x = sobel(heightmap, axis=1)
        grad_y = sobel(heightmap, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_threshold = np.percentile(grad_mag, 85)
        crater_rim_map = grad_mag > grad_threshold
        
        # Roughness Detection (Local height variance)
        window = 5
        local_mean = uniform_filter(heightmap, size=window)
        local_var = uniform_filter((heightmap - local_mean)**2, size=window)
        roughness_raw = np.sqrt(np.maximum(local_var, 0.0))
        
        # Normalize roughness 0-1
        r_min, r_max = roughness_raw.min(), roughness_raw.max()
        roughness_map = (roughness_raw - r_min) / (r_max - r_min + 1e-8)
        
        # Exaggerate roughness on crater rims
        roughness_map[crater_rim_map] = np.clip(roughness_map[crater_rim_map] + 0.2, 0, 1)
        
        return heightmap, roughness_map, crater_rim_map
