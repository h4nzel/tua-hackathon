from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import os

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Lunar Route Optimizer API"
    VERSION: str = "1.0.0"
    
    # Graph / Terrain
    GRID_SIZE: int = 100
    CELL_SIZE_METERS: float = 100.0
    MIN_ELEVATION: float = -2000.0
    MAX_ELEVATION: float = 2000.0
    
    # Cost Coefficients
    SLOPE_PENALTY: float = 0.15
    SHADOW_PENALTY: float = 2.0
    ROUGHNESS_PENALTY: float = 1.5
    HAZARD_PENALTY: float = 3.0
    CRATER_ZONE_PENALTY: float = 50.0
    CRATER_RIM_PENALTY: float = 5.0
    
    # Data Paths
    BASE_DIR: Path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    HEIGHTMAP_PATH: str = str(BASE_DIR / "full_moonfesatan.png")
    YOLO_TEST_IMAGE_PATH: str = str(BASE_DIR / "test" / "images" / "test3.png")
    YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "moon.onnx")
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"

# Instantiate singleton
settings = Settings()
