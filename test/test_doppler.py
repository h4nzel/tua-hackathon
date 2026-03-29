import numpy as np
import pycbc.noise
import pycbc.waveform
from pycbc.types import TimeSeries
from pycbc.filter import matched_filter

def calculate_anomaly_score(roughness: float, is_crater_rim: bool):
    sample_rate = 120
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Template: Normal operating frequency 10 Hz
    template_data = np.cos(2 * np.pi * 10 * t)
    template = TimeSeries(template_data, delta_t=1.0/sample_rate)
    
    # Simulate rover signal
    # If high roughness or crater rim, frequency drops (sinking/slipping) or changes amplitude
    freq = 10.0
    amp = 1.0
    
    if is_crater_rim:
        freq = 4.0 # slip
        amp = 1.5
    elif roughness > 0.6:
        freq = 10.0 - (roughness * 5.0) # sinking
        amp = 0.5
    
    signal_data = amp * np.cos(2 * np.pi * freq * t)
    
    # Add noise proportional to roughness
    noise_level = max(0.1, roughness * 0.5)
    noise_data = np.random.normal(0, noise_level, len(template_data))
    data = TimeSeries(signal_data + noise_data, delta_t=1.0/sample_rate)
    
    # Matched filter
    # To avoid psd and frequency domain complexity for simple mock, we filter directly
    # Time convolution
    snr = matched_filter(template, data, psd=None, low_frequency_cutoff=1.0)
    
    # Max SNR
    max_snr = max(abs(snr))
    
    # Normal SNR in perfect condition is approx duration * sample_rate / 2 = 60
    # Let's normalize
    norm_snr = min(1.0, max_snr / (0.5 * sample_rate * duration))
    
    # Anomaly score is inverse of match (1.0 means highly anomalous, 0.0 means normal)
    # Higher roughness also adds noise which reduces match
    anomaly = 1.0 - norm_snr
    return max(0.0, min(1.0, anomaly))

print("Normal:", calculate_anomaly_score(0.1, False))
print("High Roughness (sinking):", calculate_anomaly_score(0.8, False))
print("Crater rim (slipping):", calculate_anomaly_score(0.5, True))
