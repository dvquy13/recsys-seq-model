import pytest
from pydantic import ValidationError

from api.models import (
    ItemsByIdsRequest,
    ItemsByIdsResponse,
    PopularItemsRequest,
    RecommendationItem,
    RecommendationResponse,
    SeqRetrieverRequest,
    SeqRetrieverResponse,
)
from src.dto import RetrieveContext


def test_recommendation_item_model():
    """Test the RecommendationItem model."""
    # Test with all required fields
    item = RecommendationItem(
        score=0.9,
        main_category="Books",
        title="Test Book",
        average_rating=4.5,
        rating_number=100,
        image_url="http://example.com/image.jpg",
        parent_asin="asin123",
    )

    assert item.score == 0.9
    assert item.main_category == "Books"
    assert item.title == "Test Book"
    assert item.average_rating == 4.5
    assert item.rating_number == 100
    assert item.image_url == "http://example.com/image.jpg"
    assert item.parent_asin == "asin123"
    assert item.price is None
    assert item.subtitle is None

    # Test with optional fields
    item = RecommendationItem(
        score=0.9,
        main_category="Books",
        title="Test Book",
        average_rating=4.5,
        rating_number=100,
        price=9.99,
        subtitle="A test book",
        image_url="http://example.com/image.jpg",
        parent_asin="asin123",
    )

    assert item.price == 9.99
    assert item.subtitle == "A test book"

    # Test with missing required fields
    with pytest.raises(ValidationError):
        RecommendationItem(
            score=0.9,
            main_category="Books",
            # Missing title
            average_rating=4.5,
            rating_number=100,
            image_url="http://example.com/image.jpg",
            parent_asin="asin123",
        )


def test_recommendation_response_model():
    """Test the RecommendationResponse model."""
    # Test with minimal fields
    response = RecommendationResponse(
        recommendations=[
            RecommendationItem(
                score=0.9,
                main_category="Books",
                title="Test Book",
                average_rating=4.5,
                rating_number=100,
                image_url="http://example.com/image.jpg",
                parent_asin="asin123",
            )
        ]
    )

    assert len(response.recommendations) == 1
    assert response.ctx == {}
    assert response.debug_info is None
    assert response.metadata is None

    # Test with all fields
    response = RecommendationResponse(
        recommendations=[
            RecommendationItem(
                score=0.9,
                main_category="Books",
                title="Test Book",
                average_rating=4.5,
                rating_number=100,
                image_url="http://example.com/image.jpg",
                parent_asin="asin123",
            )
        ],
        ctx={"user_id": "user123"},
        debug_info=["Debug message 1", "Debug message 2"],
        metadata={"request_id": "req123"},
    )

    assert response.ctx == {"user_id": "user123"}
    assert response.debug_info == ["Debug message 1", "Debug message 2"]
    assert response.metadata == {"request_id": "req123"}


def test_popular_items_request_model():
    """Test the PopularItemsRequest model."""
    # Test with default values
    request = PopularItemsRequest()

    assert request.count == 10
    assert request.debug is False

    # Test with custom values
    request = PopularItemsRequest(count=5, debug=True)

    assert request.count == 5
    assert request.debug is True


def test_items_by_ids_request_model():
    """Test the ItemsByIdsRequest model."""
    # Test with required fields
    request = ItemsByIdsRequest(item_ids=["item1", "item2"])

    assert request.item_ids == ["item1", "item2"]
    assert request.debug is False

    # Test with all fields
    request = ItemsByIdsRequest(item_ids=["item1", "item2"], debug=True)

    assert request.item_ids == ["item1", "item2"]
    assert request.debug is True

    # Test with missing required fields
    with pytest.raises(ValidationError):
        ItemsByIdsRequest()


def test_items_by_ids_response_model():
    """Test the ItemsByIdsResponse model."""
    # Test with minimal fields
    response = ItemsByIdsResponse(
        items=[
            {
                "main_category": "Books",
                "title": "Test Book",
                "average_rating": 4.5,
                "rating_number": 100,
                "image_url": "http://example.com/image.jpg",
                "parent_asin": "asin123",
            }
        ]
    )

    assert len(response.items) == 1
    assert response.debug_info is None
    assert response.metadata is None

    # Test with all fields
    response = ItemsByIdsResponse(
        items=[
            {
                "main_category": "Books",
                "title": "Test Book",
                "average_rating": 4.5,
                "rating_number": 100,
                "image_url": "http://example.com/image.jpg",
                "parent_asin": "asin123",
            }
        ],
        debug_info=["Debug message 1", "Debug message 2"],
        metadata={"request_id": "req123"},
    )

    assert response.debug_info == ["Debug message 1", "Debug message 2"]
    assert response.metadata == {"request_id": "req123"}


def test_seq_retriever_request_model():
    """Test the SeqRetrieverRequest model."""
    # Test with required fields
    ctx = RetrieveContext(
        user_ids_raw=["user1"],
        item_seq_raw=[["item1", "item2"]],
        candidate_items_raw=["item3", "item4"],
    )

    request = SeqRetrieverRequest(ctx=ctx, endpoint="get_query_embeddings")

    assert request.ctx == ctx
    assert request.endpoint == "get_query_embeddings"
    assert request.debug is False

    # Test with all fields
    request = SeqRetrieverRequest(ctx=ctx, endpoint="get_query_embeddings", debug=True)

    assert request.ctx == ctx
    assert request.endpoint == "get_query_embeddings"
    assert request.debug is True

    # Test with missing required fields
    with pytest.raises(ValidationError):
        SeqRetrieverRequest(ctx=ctx)

    with pytest.raises(ValidationError):
        SeqRetrieverRequest(endpoint="get_query_embeddings")


def test_seq_retriever_response_model():
    """Test the SeqRetrieverResponse model."""
    # Test with minimal fields
    response = SeqRetrieverResponse(result={"query_embedding": [[0.1, 0.2, 0.3]]})

    assert response.result == {"query_embedding": [[0.1, 0.2, 0.3]]}
    assert response.debug_info is None
    assert response.metadata is None

    # Test with all fields
    response = SeqRetrieverResponse(
        result={"query_embedding": [[0.1, 0.2, 0.3]]},
        debug_info=["Debug message 1", "Debug message 2"],
        metadata={"request_id": "req123"},
    )

    assert response.debug_info == ["Debug message 1", "Debug message 2"]
    assert response.metadata == {"request_id": "req123"}
