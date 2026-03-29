import cv2
import numpy as np
from ultralytics import YOLO

def test_crater(image_path="test/images/test3.png", grid_size=100):
    model = YOLO('models/moon.onnx', task='detect')
    results = model(image_path)
    
    krater_haritasi = np.zeros((grid_size, grid_size), dtype=bool)
    
    # Placeholder dimensions for test. Usually derived from original image.
    orig_shape = results[0].orig_shape # (height, width)
    print(f"Original shape: {orig_shape}")
    
    for result in results:
        boxes = result.boxes
        print(f"Found {len(boxes)} craters.")
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Scale to grid_size
            gx1 = int((x1 / orig_shape[1]) * grid_size)
            gx2 = int((x2 / orig_shape[1]) * grid_size)
            gy1 = int((y1 / orig_shape[0]) * grid_size)
            gy2 = int((y2 / orig_shape[0]) * grid_size)
            
            # Ensure within bounds
            gx1, gx2 = max(0, gx1), min(grid_size-1, gx2)
            gy1, gy2 = max(0, gy1), min(grid_size-1, gy2)
            
            # Create a circular mask for the crater
            gx_center = (gx1 + gx2) // 2
            gy_center = (gy1 + gy2) // 2
            g_radius = min((gx2 - gx1) // 2, (gy2 - gy1) // 2)
            
            for y in range(gy1, gy2 + 1):
                for x in range(gx1, gx2 + 1):
                    if (x - gx_center)**2 + (y - gy_center)**2 <= (g_radius * 1.5)**2:
                        krater_haritasi[y, x] = True
    
    print(f"Total krater cells: {np.sum(krater_haritasi)}")
    return krater_haritasi

if __name__ == "__main__":
    test_crater()
