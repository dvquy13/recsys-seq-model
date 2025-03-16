import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# We need to patch several dependencies before importing app
with (
    patch("api.app.init_id_mapper") as mock_init_id_mapper,
    patch("api.app.ConfigLoader") as mock_config_loader,
    patch("api.app.redis.Redis") as mock_redis,
    patch("api.app.QdrantClient") as mock_qdrant,
):
    # Mock IDMapper
    mock_idm = MagicMock()
    mock_idm.get_item_index.side_effect = (
        lambda x: int(x.replace("item", "")) if x.startswith("item") else 1
    )
    mock_init_id_mapper.return_value = mock_idm

    # Mock ConfigLoader
    mock_cfg = MagicMock()
    mock_cfg.redis.host = "localhost"
    mock_cfg.redis.port = 6379
    mock_cfg.redis.keys.recent_key_prefix = "recent_"
    mock_cfg.redis.keys.popular_key = "popular"
    mock_cfg.vectorstore.qdrant.url = "http://localhost:6333"
    mock_cfg.vectorstore.qdrant.collection_name = "test_collection"
    mock_cfg.data.train_features_fp = "./test_features.json"
    mock_config_loader.return_value = mock_cfg

    # Mock Redis client
    mock_redis_client = MagicMock()
    mock_redis.return_value = mock_redis_client

    # Mock Qdrant client
    mock_qdrant_client = MagicMock()
    mock_qdrant.return_value = mock_qdrant_client

    # Now import the app
    from api.app import app, get_recommendation_service, get_services
    from api.models import (
        ItemsByIdsResponse,
        RecommendationItem,
        RecommendationResponse,
        SearchByTitleResponse,
        SeqRetrieverResponse,
    )


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = MagicMock()

    # Set up mock data for popular items
    popular_items_data = {
        "rec_item_ids": ["item1", "item2", "item3"],
        "rec_scores": [0.9, 0.8, 0.7],
    }
    mock_client.get.return_value = json.dumps(popular_items_data)

    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = MagicMock()

    # Mock search results
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

    mock_client.search.return_value = [mock_hit1, mock_hit2]
    mock_client.retrieve.return_value = [mock_hit1, mock_hit2]

    # Mock scroll results for text search
    mock_scroll_hit1 = MagicMock()
    mock_scroll_hit1.payload = {
        "main_category": "Books",
        "title": "Harry Potter and the Philosopher's Stone",
        "average_rating": 4.7,
        "rating_number": 5000,
        "image_url": "http://example.com/harry_potter.jpg",
        "parent_asin": "asin1001",
    }

    mock_scroll_hit2 = MagicMock()
    mock_scroll_hit2.payload = {
        "main_category": "Books",
        "title": "Harry Potter and the Chamber of Secrets",
        "average_rating": 4.6,
        "rating_number": 4800,
        "image_url": "http://example.com/harry_potter2.jpg",
        "parent_asin": "asin1002",
    }

    mock_client.scroll.return_value = ([mock_scroll_hit1, mock_scroll_hit2], None)

    return mock_client


@pytest.fixture
def mock_idm():
    """Mock ID mapper for testing."""
    mock_mapper = MagicMock()
    mock_mapper.get_item_index.side_effect = (
        lambda x: int(x.replace("item", "")) if x.startswith("item") else 1
    )
    return mock_mapper


@pytest.fixture
def mock_services(mock_redis_client, mock_qdrant_client, mock_idm):
    """Create mock services for testing."""
    with patch("api.app.Services") as MockServices:
        mock_services = MockServices.return_value
        mock_services.redis_client = mock_redis_client
        mock_services.ann_index = mock_qdrant_client
        mock_services.idm = mock_idm
        yield mock_services


@pytest.fixture
def mock_recommendation_service():
    """Create a mock recommendation service with predefined responses."""
    mock_service = MagicMock()

    # Mock get_popular_recommendations
    popular_response = RecommendationResponse(
        recommendations=[
            RecommendationItem(
                score=0.9,
                main_category="Books",
                title="Test Book 1",
                average_rating=4.5,
                rating_number=100,
                image_url="http://example.com/image1.jpg",
                parent_asin="asin1",
            ),
            RecommendationItem(
                score=0.8,
                main_category="Books",
                title="Test Book 2",
                average_rating=4.2,
                rating_number=80,
                image_url="http://example.com/image2.jpg",
                parent_asin="asin2",
            ),
        ],
        ctx={},
    )
    mock_service.get_popular_recommendations = AsyncMock(return_value=popular_response)

    # Mock retrieve_recommendations
    mock_service.retrieve_recommendations = AsyncMock(return_value=popular_response)

    # Mock get_items_by_ids
    items_response = ItemsByIdsResponse(
        items=[
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
    )
    mock_service.get_items_by_ids = AsyncMock(return_value=items_response)

    # Mock call_seq_retriever
    seq_retriever_response = SeqRetrieverResponse(
        result={"query_embedding": [[0.1, 0.2, 0.3]]}
    )
    mock_service.call_seq_retriever = AsyncMock(return_value=seq_retriever_response)

    # Mock search_items_by_title
    search_response = SearchByTitleResponse(
        items=[
            RecommendationItem(
                score=1.0,
                main_category="Books",
                title="Harry Potter and the Philosopher's Stone",
                average_rating=4.7,
                rating_number=5000,
                image_url="http://example.com/harry_potter.jpg",
                parent_asin="asin1001",
            ),
            RecommendationItem(
                score=1.0,
                main_category="Books",
                title="Harry Potter and the Chamber of Secrets",
                average_rating=4.6,
                rating_number=4800,
                image_url="http://example.com/harry_potter2.jpg",
                parent_asin="asin1002",
            ),
        ],
        debug_info=["Title search query: Harry Potter", "Results count: 2"],
    )
    mock_service.search_items_by_title = AsyncMock(return_value=search_response)

    return mock_service


@pytest.fixture
def client(mock_services, mock_recommendation_service):
    """Create a test client for the FastAPI app."""
    app.dependency_overrides[get_services] = lambda: mock_services
    app.dependency_overrides[get_recommendation_service] = (
        lambda: mock_recommendation_service
    )

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides = {}
