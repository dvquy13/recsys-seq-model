from typing import List, Optional, Union

from pydantic import BaseModel

from src.dto import RetrieveContext


class RecommendationItem(BaseModel):
    score: float
    main_category: str
    title: str
    average_rating: float
    rating_number: int
    price: Optional[Union[str, float]]
    subtitle: str
    image_url: str
    parent_asin: str


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    ctx: RetrieveContext
