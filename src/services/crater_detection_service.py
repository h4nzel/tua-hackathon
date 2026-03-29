import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from src.core.config import settings

logger = logging.getLogger("LunarRouter.CraterService")

class CraterDetectionService:
    """Service handling computer vision logic to detect lunar craters as no-go zones."""
    
    def __init__(self):
        self.grid_size = settings.GRID_SIZE
        self.model_path = settings.YOLO_MODEL_PATH
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(f"ONNX Model not found at {self.model_path}. Crater detection will return empty array.")
            return
            
        try:
            # CPU optimized load
            self.model = YOLO(self.model_path, task="detect")
            logger.info(f"Loaded ONNX model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.model = None

    def detect_craters(self, image_path: str) -> np.ndarray:
        """
        Runs inference on given image and translates bounding boxes to a grid_size boolean map.
        """
        crater_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        if self.model is None or not os.path.exists(image_path):
            return crater_map
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image at {image_path}")
                return crater_map
                
            orig_h, orig_w = img.shape[:2]
            results = self.model(img, verbose=False) # Run inference
            
            total_craters = 0
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # Convert pixel coordinates to grid scale
                    grid_x1 = int((x1 / orig_w) * self.grid_size)
                    grid_x2 = int((x2 / orig_w) * self.grid_size)
                    grid_y1 = int((y1 / orig_h) * self.grid_size)
                    grid_y2 = int((y2 / orig_h) * self.grid_size)
                    
                    # Ensure coordinates are safely within grid
                    grid_x1, grid_x2 = sorted([np.clip(grid_x1, 0, self.grid_size-1), 
                                                np.clip(grid_x2, 0, self.grid_size-1)])
                    grid_y1, grid_y2 = sorted([np.clip(grid_y1, 0, self.grid_size-1), 
                                                np.clip(grid_y2, 0, self.grid_size-1)])
                    
                    cx = (grid_x1 + grid_x2) / 2
                    cy = (grid_y1 + grid_y2) / 2
                    radius = max(grid_x2 - grid_x1, grid_y2 - grid_y1) / 2.0
                    
                    # Optional: Add small margin of error (buffer)
                    radius = max(1, radius * 1.1)
                    
                    # Create circular mask in the bounding box
                    y, x = np.ogrid[-int(cy):self.grid_size-int(cy), -int(cx):self.grid_size-int(cx)]
                    mask = (x*x + y*y <= radius*radius)
                    
                    crater_map[mask] = True
                    total_craters += 1
            
            logger.info(f"Detected {total_craters} craters mapping to {np.sum(crater_map)} grid blocks.")            
            return crater_map
            
        except Exception as e:
            logger.error(f"Error during crater detection inference: {e}")
            return crater_map
