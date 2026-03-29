import httpx
from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

response = client.post(
    "/api/v1/chat/message",
    data={"prompt": "Hey! Draw a route from 10,10 to 90,90"}
)

print(response.status_code)
print(response.json())
