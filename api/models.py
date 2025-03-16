from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.dto import RetrieveContext


class RecommendationItem(BaseModel):
    score: float
    main_category: str
    title: str
    average_rating: float
    rating_number: int
    price: Optional[Union[str, float]] = None
    subtitle: Optional[str] = None
    image_url: str
    parent_asin: str


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    ctx: Dict[str, Any] = {}
    debug_info: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PopularItemsRequest(BaseModel):
    count: int = Field(10, description="Number of popular items to return")
    debug: bool = Field(False, description="Enable debug logging")


class ItemsByIdsRequest(BaseModel):
    item_ids: List[str] = Field(..., description="List of item IDs to retrieve")
    debug: bool = Field(False, description="Enable debug logging")


class ItemsByIdsResponse(BaseModel):
    items: List[Dict[str, Any]]
    debug_info: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SeqRetrieverRequest(BaseModel):
    ctx: RetrieveContext
    endpoint: str = Field(..., description="Endpoint to call on the model server")
    debug: bool = Field(False, description="Enable debug logging")


class SeqRetrieverResponse(BaseModel):
    result: Dict[str, Any]
    debug_info: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchByTitleRequest(BaseModel):
    query: str = Field(..., description="Title search query")
    limit: int = Field(10, description="Maximum number of results to return")
    debug: bool = Field(False, description="Enable debug logging")


class SearchByTitleResponse(BaseModel):
    items: List[RecommendationItem]
    debug_info: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
