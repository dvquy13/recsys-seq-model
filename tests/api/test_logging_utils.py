import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from starlette.responses import Response
from starlette.testclient import TestClient

from api.logging_utils import RequestIDMiddleware


@pytest.fixture
def app():
    """Create a test FastAPI app with the RequestIDMiddleware."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test-json")
    async def test_json():
        return {"message": "test", "metadata": None}

    @app.get("/test-json-with-metadata")
    async def test_json_with_metadata():
        return {"message": "test", "metadata": {"existing": "data"}}

    @app.get("/test-non-json")
    async def test_non_json():
        return Response(content="test", media_type="text/plain")

    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return TestClient(app)


def test_request_id_middleware_json_response(client):
    """Test that the middleware adds a rec_id to JSON responses."""
    response = client.get("/test-json")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "rec_id" in data["metadata"]
    assert uuid.UUID(data["metadata"]["rec_id"])  # Verify it's a valid UUID


def test_request_id_middleware_json_response_with_existing_metadata(client):
    """Test that the middleware adds a rec_id to JSON responses with existing metadata."""
    response = client.get("/test-json-with-metadata")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "existing" in data["metadata"]
    assert "rec_id" in data["metadata"]
    assert data["metadata"]["existing"] == "data"
    assert uuid.UUID(data["metadata"]["rec_id"])  # Verify it's a valid UUID


def test_request_id_middleware_non_json_response(client):
    """Test that the middleware doesn't modify non-JSON responses."""
    response = client.get("/test-non-json")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert response.text == "test"


@pytest.mark.asyncio
async def test_request_id_middleware_dispatch():
    """Test the basic functionality of the middleware dispatch method."""
    # Create a mock request
    request = MagicMock(spec=Request)
    request.state = MagicMock()

    # Create a mock response (non-streaming)
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "text/plain"}

    # Create the middleware
    middleware = RequestIDMiddleware(app=None)

    # Mock the call_next function
    call_next = AsyncMock(return_value=mock_response)

    # Call the middleware
    with patch(
        "uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")
    ):
        response = await middleware.dispatch(request, call_next)

    # Check that the request ID was added to the request state
    assert request.state.rec_id == "12345678-1234-5678-1234-567812345678"

    # Check that the response was returned unchanged for non-JSON responses
    assert response == mock_response
