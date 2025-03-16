import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.services import RecommendationService
from src.dto import RetrieveContext


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    services = MagicMock()

    # Mock Redis client
    services.redis_client = MagicMock()
    popular_items_data = {
        "rec_item_ids": ["item1", "item2", "item3"],
        "rec_scores": [0.9, 0.8, 0.7],
    }
    services.redis_client.get.return_value = json.dumps(popular_items_data)

    # Mock Qdrant client
    services.ann_index = MagicMock()
    mock_hit1 = MagicMock()
    mock_hit1.model_dump.return_value = {"score": 0.9}
    mock_hit1.payload = {
        "main_category": "Books",
        "title": "Test Book 1",
        "average_rating": 4.5,
        "rating_number": 100,
        "image_url": "http://example.com/image1.jpg",
        "parent_asin": "asin1",
    }

    mock_hit2 = MagicMock()
    mock_hit2.model_dump.return_value = {"score": 0.8}
    mock_hit2.payload = {
        "main_category": "Books",
        "title": "Test Book 2",
        "average_rating": 4.2,
        "rating_number": 80,
        "image_url": "http://example.com/image2.jpg",
        "parent_asin": "asin2",
    }

    services.ann_index.search.return_value = [mock_hit1, mock_hit2]
    services.ann_index.retrieve.return_value = [mock_hit1, mock_hit2]

    # Mock ID mapper
    services.idm = MagicMock()
    services.idm.get_item_index.side_effect = (
        lambda x: int(x.replace("item", "")) if x.startswith("item") else 1
    )

    return services


@pytest.fixture
def mock_cfg():
    """Create a mock configuration."""
    mock_cfg = MagicMock()
    mock_cfg.redis.keys.recent_key_prefix = "recent_"
    mock_cfg.redis.keys.popular_key = "popular"
    mock_cfg.vectorstore.qdrant.collection_name = "test_collection"
    return mock_cfg


@pytest.fixture
def recommendation_service(mock_services, mock_cfg):
    """Create a recommendation service with mocked dependencies."""
    with (
        patch("api.services.cfg", mock_cfg),
        patch("api.services.os.getenv", return_value="http://localhost:3000"),
        patch("api.services.logger", MagicMock()),  # Mock logger to avoid rec_id errors
    ):
        service = RecommendationService(mock_services)
        return service


@pytest.mark.asyncio
async def test_get_recommendations_from_redis(recommendation_service, mock_services):
    """Test getting recommendations from Redis."""
    result = recommendation_service.get_recommendations_from_redis("popular", 2)

    # Check that Redis was called with the correct key
    mock_services.redis_client.get.assert_called_once_with("popular")

    # Check the result
    assert result["rec_item_ids"] == ["item1", "item2"]
    assert result["rec_scores"] == [0.9, 0.8]


@pytest.mark.asyncio
async def test_get_recommendations_from_redis_not_found(
    recommendation_service, mock_services
):
    """Test getting recommendations from Redis when the key is not found."""
    mock_services.redis_client.get.return_value = None

    with pytest.raises(HTTPException) as excinfo:
        recommendation_service.get_recommendations_from_redis("nonexistent", 2)

    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_get_user_prev_interactions(recommendation_service, mock_services):
    """Test getting user previous interactions."""
    mock_services.redis_client.get.return_value = "item1__item2__item3"

    result = recommendation_service.get_user_prev_interactions("user1")

    # Check that Redis was called with the correct key
    mock_services.redis_client.get.assert_called_once_with("recent_user1")

    # Check the result
    assert result["recent_interactions"] == ["item1", "item2", "item3"]


@pytest.mark.asyncio
async def test_get_user_prev_interactions_not_found(
    recommendation_service, mock_services
):
    """Test getting user previous interactions when the user is not found."""
    mock_services.redis_client.get.return_value = None

    result = recommendation_service.get_user_prev_interactions("nonexistent")

    # Check the result
    assert result["recent_interactions"] == []


@pytest.mark.asyncio
async def test_get_items_by_ids(recommendation_service, mock_services, mock_cfg):
    """Test getting items by IDs."""
    # Patch the collection_name in the service call
    with patch.object(
        mock_cfg.vectorstore.qdrant, "collection_name", "test_collection"
    ):
        result = await recommendation_service.get_items_by_ids(["item1", "item2"])

        # Check that the ID mapper was called with the correct IDs
        assert mock_services.idm.get_item_index.call_count == 2
        mock_services.idm.get_item_index.assert_any_call("item1")
        mock_services.idm.get_item_index.assert_any_call("item2")

        # Check that the vector store was called with the correct indices
        mock_services.ann_index.retrieve.assert_called_once()
        call_args = mock_services.ann_index.retrieve.call_args[1]
        assert call_args["ids"] == [1, 2]

        # Check the result
        assert len(result.items) == 2
        assert result.items[0]["title"] == "Test Book 1"
        assert result.items[1]["title"] == "Test Book 2"


@pytest.mark.asyncio
async def test_get_popular_recommendations(recommendation_service, mock_services):
    """Test getting popular recommendations."""
    # Mock the get_items_by_ids method
    recommendation_service.get_items_by_ids = AsyncMock()
    recommendation_service.get_items_by_ids.return_value.items = [
        {
            "main_category": "Books",
            "title": "Test Book 1",
            "average_rating": 4.5,
            "rating_number": 100,
            "image_url": "http://example.com/image1.jpg",
            "parent_asin": "asin1",
        },
        {
            "main_category": "Books",
            "title": "Test Book 2",
            "average_rating": 4.2,
            "rating_number": 80,
            "image_url": "http://example.com/image2.jpg",
            "parent_asin": "asin2",
        },
    ]

    result = await recommendation_service.get_popular_recommendations(2)

    # Check that the get_recommendations_from_redis method was called
    mock_services.redis_client.get.assert_called_once_with("popular")

    # Check that the get_items_by_ids method was called with the correct IDs
    recommendation_service.get_items_by_ids.assert_called_once_with(["item1", "item2"])

    # Check the result
    assert len(result.recommendations) == 2
    assert result.recommendations[0].title == "Test Book 1"
    assert result.recommendations[0].score == 0.9
    assert result.recommendations[1].title == "Test Book 2"
    assert result.recommendations[1].score == 0.8


@pytest.mark.asyncio
async def test_retrieve_recommendations(
    recommendation_service, mock_services, mock_cfg
):
    """Test retrieving recommendations."""
    # Mock the call_seq_retriever method
    recommendation_service.call_seq_retriever = AsyncMock()
    recommendation_service.call_seq_retriever.return_value.result = {
        "query_embedding": [[0.1, 0.2, 0.3]]
    }

    # Create a test context
    ctx = RetrieveContext(
        user_ids_raw=["user1"],
        item_seq_raw=[["item1", "item2"]],
        candidate_items_raw=["item3", "item4"],
    )

    # Patch the collection_name in the service call
    with patch.object(
        mock_cfg.vectorstore.qdrant, "collection_name", "test_collection"
    ):
        result = await recommendation_service.retrieve_recommendations(ctx, 2)

        # Check that the call_seq_retriever method was called with the correct parameters
        recommendation_service.call_seq_retriever.assert_called_once_with(
            ctx, "get_query_embeddings"
        )

        # Check that the vector store was called with the correct parameters
        mock_services.ann_index.search.assert_called_once()

        # Check the result
        assert len(result.recommendations) == 2
        assert result.recommendations[0].title == "Test Book 1"
        assert result.recommendations[0].score == 0.9
        assert result.recommendations[1].title == "Test Book 2"
        assert result.recommendations[1].score == 0.8


@pytest.mark.asyncio
async def test_retrieve_recommendations_empty_context(recommendation_service):
    """Test retrieving recommendations with an empty context."""
    # Mock the get_popular_recommendations method
    recommendation_service.get_popular_recommendations = AsyncMock()

    # Create an empty context
    ctx = RetrieveContext(user_ids_raw=[], item_seq_raw=[[]], candidate_items_raw=[])

    await recommendation_service.retrieve_recommendations(ctx, 2)

    # Check that the get_popular_recommendations method was called
    recommendation_service.get_popular_recommendations.assert_called_once_with(2)


@pytest.mark.asyncio
async def test_call_seq_retriever(recommendation_service):
    """Test calling the sequence retriever."""
    # Mock the httpx.AsyncClient
    with (
        patch("api.services.httpx.AsyncClient") as mock_client_class,
        patch("api.services.json.dumps", lambda x: '{"mocked": "json"}'),
    ):  # Mock json.dumps to avoid serialization issues
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock the response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        # Use a regular dict instead of a coroutine for json()
        mock_response.json = MagicMock(
            return_value={"query_embedding": [[0.1, 0.2, 0.3]]}
        )
        mock_client.post.return_value = mock_response

        # Create a test context
        ctx = RetrieveContext(
            user_ids_raw=["user1"],
            item_seq_raw=[["item1", "item2"]],
            candidate_items_raw=["item3", "item4"],
        )

        result = await recommendation_service.call_seq_retriever(
            ctx, "get_query_embeddings"
        )

        # Check that the httpx client was called with the correct parameters
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args[0]
        assert "get_query_embeddings" in call_args[0]

        # Check the result
        assert result.result["query_embedding"] == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_call_seq_retriever_error(recommendation_service):
    """Test calling the sequence retriever when an error occurs."""
    # Mock the httpx.AsyncClient
    with (
        patch("api.services.httpx.AsyncClient") as mock_client_class,
        patch("api.services.json.dumps", lambda x: '{"mocked": "json"}'),
    ):  # Mock json.dumps to avoid serialization issues
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock the response with an error
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        # Use a regular MagicMock for json() to avoid coroutine issues
        mock_response.json = MagicMock(return_value={"error": "Internal Server Error"})
        mock_client.post.return_value = mock_response

        # Create a test context
        ctx = RetrieveContext(
            user_ids_raw=["user1"],
            item_seq_raw=[["item1", "item2"]],
            candidate_items_raw=["item3", "item4"],
        )

        with pytest.raises(HTTPException) as excinfo:
            await recommendation_service.call_seq_retriever(ctx, "get_query_embeddings")

        assert excinfo.value.status_code == 500
