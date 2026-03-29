from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

response = client.post(
    "/api/v1/routes/calculate",
    json={
        "start": {"x": 10, "y": 10},
        "target": {"x": 90, "y": 90},
        "time_offset_hours": 0.0,
        "algorithm": "dl-alt"
    }
)

print(response.status_code)
print(response.json())
