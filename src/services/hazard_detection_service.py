import numpy as np
import logging
from src.core.config import settings

logger = logging.getLogger("LunarRouter.HazardService")

# Conditional PyCBC import
try:
    from pycbc.types import TimeSeries
    from pycbc.filter import matched_filter
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False
    logger.warning("PyCBC disabled or not installed. Hazard map will default to 0.0.")

class HazardDetectionService:
    """Service to process rover wheel telemetry for anomaly detection (sinking/slipping)."""
    
    def __init__(self):
        self.grid_size = settings.GRID_SIZE
        self.sample_rate = 100
        self.duration = 2.0
        
        # Base templates
        if PYCBC_AVAILABLE:
            t = np.arange(0, self.duration, 1.0 / self.sample_rate)
            self._nominal_freq = 10.0
            
            # Simulated ideal flat terrain wheel rotation
            pure_sine = np.sin(2 * np.pi * self._nominal_freq * t)
            
            # Simulated edge case templates
            self.sink_template = TimeSeries(pure_sine * np.exp(-t * 2), delta_t=1.0/self.sample_rate) # Viscous regolith sinkage (damping)
            self.slip_template = TimeSeries(pure_sine * (1 + 0.5 * np.sin(2*np.pi*25*t)), delta_t=1.0/self.sample_rate) # Crater boundary slipping (high freq jitter)

    def analyze_node_hazard(self, roughness: float, is_crater_rim: bool) -> float:
        """
        Simulates telemetry and runs matched filtering for a single grid node.
        0.0 = Safe, 1.0 = Highly hazardous (stuck or spinning out)
        """
        if not PYCBC_AVAILABLE:
            return 0.0
            
        t = np.arange(0, self.duration, 1.0 / self.sample_rate)
        
        # 1. Simulate data based on terrain factors
        if is_crater_rim:
            # Slipping scenario
            freq = self._nominal_freq * (1.0 + roughness * 0.5)
            noise_amp = 0.3 + roughness
            data = np.sin(2 * np.pi * freq * t) + np.random.normal(0, noise_amp, len(t))
            target_template = self.slip_template
        else:
            # Sinking scenario based on roughness
            freq = max(1.0, self._nominal_freq * (1.0 - roughness * 0.8)) # Drops frequency
            damping = np.exp(-t * roughness * 3)
            noise_amp = 0.1 + roughness * 0.5
            data = np.sin(2 * np.pi * freq * t) * damping + np.random.normal(0, noise_amp, len(t))
            target_template = self.sink_template
            
        ts_data = TimeSeries(data, delta_t=1.0/self.sample_rate)
        
        # 2. Matched filter execution
        try:
            snr = matched_filter(target_template, ts_data, psd=None, low_frequency_cutoff=1.0)
            peak_snr = float(max(abs(snr)))
            
            # 3. Decision translation
            hazard_score = 0.0
            if is_crater_rim:
                hazard_score = np.clip((peak_snr - 10.0) / 20.0, 0.0, 1.0)
            else:
                hazard_score = np.clip((peak_snr - 8.0) / 15.0, 0.0, 1.0) * (roughness + 0.1)
                
            return float(hazard_score)
        except Exception as e:
            return 0.0

    def create_hazard_map(self, roughness_map: np.ndarray, crater_rim_map: np.ndarray) -> np.ndarray:
        """Generates the full hazard map for the entire grid."""
        logger.info(f"Generating Doppler Hazard Map (PyCBC) for ({self.grid_size}x{self.grid_size}) grid.")
        hazard_map = np.zeros((self.grid_size, self.grid_size))
        
        if not PYCBC_AVAILABLE:
            return hazard_map
            
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if roughness_map[y, x] > 0.05 or crater_rim_map[y, x]:
                    hazard_map[y, x] = self.analyze_node_hazard(roughness_map[y, x], bool(crater_rim_map[y, x]))
        
        logger.info(f"Hazard scores range: {hazard_map.min():.2f} — {hazard_map.max():.2f}")
        return hazard_map
