import pytest
import httpx
from app.main import app

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.anyio
async def test_health_check():
    """
    Verify the health endpoint is working correctly.
    """
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

@pytest.mark.anyio
async def test_chat_endpoint_schema():
    """
    Verify the chat endpoint accepts requests and returns the correct response schema.
    """
    payload = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify strict schema compliance
        assert "reply" in data
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert "end_of_conversation" in data
        assert isinstance(data["end_of_conversation"], bool)

@pytest.mark.anyio
async def test_chat_invalid_payload():
    """
    Verify the chat endpoint rejects invalid payloads.
    """
    payload = {
        "invalid_key": "Invalid key"
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/chat", json=payload)
        assert response.status_code == 422  # Unprocessable Entity
