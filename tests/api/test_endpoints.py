import json
from unittest.mock import patch

import pytest

from src.dto import RetrieveContext


def test_get_popular_recommendations(client, mock_recommendation_service):
    """Test the popular recommendations endpoint."""
    response = client.get("/recs/popular")

    assert response.status_code == 200
    data = response.json()

    # Check that the response has the expected structure
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2

    # Check that the mock service was called
    mock_recommendation_service.get_popular_recommendations.assert_called_once_with(10)

    # Check the content of the recommendations
    recommendations = data["recommendations"]
    assert recommendations[0]["title"] == "Test Book 1"
    assert recommendations[0]["score"] == 0.9
    assert recommendations[1]["title"] == "Test Book 2"
    assert recommendations[1]["score"] == 0.8


def test_get_popular_recommendations_with_count(client, mock_recommendation_service):
    """Test the popular recommendations endpoint with a custom count."""
    response = client.get("/recs/popular?count=5")

    assert response.status_code == 200

    # Check that the mock service was called with the correct count
    mock_recommendation_service.get_popular_recommendations.assert_called_once_with(5)


def test_retrieve_recommendations(client, mock_recommendation_service):
    """Test the retrieve recommendations endpoint."""
    # Create a test context
    ctx = RetrieveContext(
        user_ids_raw=["user1"],
        item_seq_raw=[["item1", "item2"]],
        candidate_items_raw=["item3", "item4"],
    )

    response = client.post(
        "/recs/retrieve",
        json=ctx.model_dump(),
    )

    assert response.status_code == 200
    data = response.json()

    # Check that the response has the expected structure
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2

    # Check that the mock service was called with the correct parameters
    mock_recommendation_service.retrieve_recommendations.assert_called_once()
    call_args = mock_recommendation_service.retrieve_recommendations.call_args[0]
    assert isinstance(call_args[0], RetrieveContext)
    assert call_args[0].user_ids_raw == ["user1"]
    assert call_args[0].item_seq_raw == [["item1", "item2"]]
    assert call_args[0].candidate_items_raw == ["item3", "item4"]
    assert call_args[1] == 10  # Default count


def test_retrieve_recommendations_with_count(client, mock_recommendation_service):
    """Test the retrieve recommendations endpoint with a custom count."""
    # Create a test context
    ctx = RetrieveContext(
        user_ids_raw=["user1"],
        item_seq_raw=[["item1", "item2"]],
        candidate_items_raw=["item3", "item4"],
    )

    response = client.post(
        "/recs/retrieve?count=5",
        json=ctx.model_dump(),
    )

    assert response.status_code == 200

    # Check that the mock service was called with the correct count
    mock_recommendation_service.retrieve_recommendations.assert_called_once()
    call_args = mock_recommendation_service.retrieve_recommendations.call_args[0]
    assert call_args[1] == 5


def test_get_items_by_ids(client, mock_recommendation_service):
    """Test the get items by IDs endpoint."""
    request_data = {"item_ids": ["item1", "item2"], "debug": False}

    response = client.post(
        "/items/get_by_ids",
        json=request_data,
    )

    assert response.status_code == 200
    data = response.json()

    # Check that the response has the expected structure
    assert "items" in data
    assert len(data["items"]) == 2

    # Check that the mock service was called with the correct parameters
    mock_recommendation_service.get_items_by_ids.assert_called_once_with(
        ["item1", "item2"]
    )

    # Check the content of the items
    items = data["items"]
    assert items[0]["title"] == "Test Book 1"
    assert items[1]["title"] == "Test Book 2"


def test_seq_retriever(client, mock_recommendation_service):
    """Test the sequence retriever endpoint."""
    # Create a test context
    ctx = RetrieveContext(
        user_ids_raw=["user1"],
        item_seq_raw=[["item1", "item2"]],
        candidate_items_raw=["item3", "item4"],
    )

    request_data = {
        "ctx": ctx.model_dump(),
        "endpoint": "get_query_embeddings",
        "debug": False,
    }

    response = client.post(
        "/vendor/seq_retriever",
        json=request_data,
    )

    assert response.status_code == 200
    data = response.json()

    # Check that the response has the expected structure
    assert "result" in data
    assert "query_embedding" in data["result"]

    # Check that the mock service was called with the correct parameters
    mock_recommendation_service.call_seq_retriever.assert_called_once()
    call_args = mock_recommendation_service.call_seq_retriever.call_args[0]
    assert isinstance(call_args[0], RetrieveContext)
    assert call_args[0].user_ids_raw == ["user1"]
    assert call_args[0].item_seq_raw == [["item1", "item2"]]
    assert call_args[0].candidate_items_raw == ["item3", "item4"]
    assert call_args[1] == "get_query_embeddings"
