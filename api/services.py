import json
import os
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import HTTPException
from loguru import logger
from qdrant_client.http.models import FieldCondition, Filter, MatchText

from src.cfg import ConfigLoader
from src.dto import RetrieveContext

from .models import (
    ItemsByIdsResponse,
    RecommendationItem,
    RecommendationResponse,
    SearchByTitleResponse,
    SeqRetrieverResponse,
)

cfg = ConfigLoader("./cfg/common.yaml")


class RecommendationService:
    def __init__(self, services):
        self.services = services
        self.redis_feature_prev_items_key_prefix = cfg.redis.keys.recent_key_prefix
        self.redis_output_popular_key = cfg.redis.keys.popular_key
        self.seq_retriever_model_server_url = os.getenv(
            "SEQ_RETRIEVER_MODEL_SERVER_URL", "http://localhost:3000"
        )

    def get_recommendations_from_redis(
        self, redis_key: str, count: Optional[int]
    ) -> Dict[str, Any]:
        rec_data = self.services.redis_client.get(redis_key)
        if not rec_data:
            error_message = f"[DEBUG] No recommendations found for key: {redis_key}"
            logger.error(error_message)
            raise HTTPException(status_code=404, detail=error_message)
        rec_data_json = json.loads(rec_data)
        rec_item_ids = rec_data_json.get("rec_item_ids", [])
        rec_scores = rec_data_json.get("rec_scores", [])
        if count is not None:
            rec_item_ids = rec_item_ids[:count]
            rec_scores = rec_scores[:count]
        return {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores}

    def get_user_prev_interactions(self, user_id: str) -> Dict[str, Any]:
        key = self.redis_feature_prev_items_key_prefix + user_id
        data = self.services.redis_client.get(key)
        if not data:
            error_message = f"[DEBUG] No recommendations found for key: {key}"
            logger.error(error_message)
            return {"recent_interactions": []}
        return {"recent_interactions": data.split("__")}

    async def get_items_by_ids(self, item_ids: List[str]) -> ItemsByIdsResponse:
        """
        Retrieve items by their IDs. The IDs will be mapped to indices before querying the vector store.
        """
        # Map string IDs to indices
        indices = [self.services.idm.get_item_index(item_id) for item_id in item_ids]
        logger.info(f"[DEBUG] Mapped item IDs {item_ids} to indices {indices}")

        # Retrieve items from vector store
        hits = self.services.ann_index.retrieve(
            collection_name=cfg.vectorstore.qdrant.collection_name,
            ids=indices,
        )
        outputs = [hit.payload for hit in hits]

        return ItemsByIdsResponse(items=outputs)

    async def get_popular_recommendations(self, count: int) -> RecommendationResponse:
        recommendations = self.get_recommendations_from_redis(
            self.redis_output_popular_key, count
        )
        rec_item_ids = recommendations["rec_item_ids"]

        item_info = await self.get_items_by_ids(rec_item_ids)

        for i, item in enumerate(item_info.items):
            item["score"] = recommendations["rec_scores"][i]

        logger.info(f"[DEBUG] {item_info=}")
        return RecommendationResponse(recommendations=item_info.items, ctx={})

    async def retrieve_recommendations(
        self, ctx: RetrieveContext, count: int
    ) -> RecommendationResponse:
        # Items to exclude from recommendations
        items_to_exclude = set()

        if len(ctx.user_ids_raw) > 0 and (user_id := ctx.user_ids_raw[0]):
            logger.info(f"Getting recent interactions for user: {user_id}")
            user_id = ctx.user_ids_raw[0]
            user_prev_interactions = self.get_user_prev_interactions(user_id)[
                "recent_interactions"
            ]
            logger.info(f"[DEBUG] {user_prev_interactions=}")

            # Add user's previous interactions to exclusion set
            items_to_exclude.update(user_prev_interactions)

            curr_item_seq = ctx.item_seq_raw[0]
            ctx.item_seq_raw = [user_prev_interactions + curr_item_seq]
            logger.info(f"[DEBUG] {ctx=}")

        # Add items from input sequence to exclusion set
        if ctx.item_seq_raw and ctx.item_seq_raw[0]:
            items_to_exclude.update(ctx.item_seq_raw[0])

        logger.info(
            f"[DEBUG] Items to exclude from recommendations: {items_to_exclude}"
        )

        if len(ctx.item_seq_raw[0]) == 0:
            logger.info("Empty RetrieveContext, fallback to popular recommendations")
            return await self.get_popular_recommendations(count)

        query_embedding_resp = await self.call_seq_retriever(
            ctx, "get_query_embeddings"
        )
        query_embedding = np.array(query_embedding_resp.result["query_embedding"])
        logger.info(f"[DEBUG] {query_embedding.shape=}")

        # Get more recommendations than needed since we'll filter some out
        buffer_count = count + len(items_to_exclude)
        hits = self.services.ann_index.search(
            collection_name=cfg.vectorstore.qdrant.collection_name,
            query_vector=query_embedding[0],
            limit=buffer_count,
        )

        # Filter out items that should be excluded
        filtered_recommendations = []
        for hit in hits:
            # TODO: This knowledge of using parent_asin as item id should be clear to developers...
            item_id = hit.payload.get("parent_asin", "")
            if item_id not in items_to_exclude:
                filtered_recommendations.append(
                    {"score": hit.model_dump()["score"], **hit.payload}
                )
                if len(filtered_recommendations) >= count:
                    break

        return RecommendationResponse(
            recommendations=filtered_recommendations,
            ctx=ctx.model_dump(),
            debug_info=query_embedding_resp.debug_info,
        )

    async def call_seq_retriever(
        self, ctx: RetrieveContext, endpoint: str
    ) -> SeqRetrieverResponse:
        user_ids = ctx.user_ids_raw
        item_seq = ctx.item_seq_raw
        candidate_items = ctx.candidate_items_raw

        logger.debug(
            f"Calling seq_rating_predicting with user_ids: {user_ids}, item_seq: {item_seq} and item_ids: {candidate_items}"
        )

        # Prepare the payload for the external service
        payload = {"ctx": ctx.model_dump()}

        # Using json.dumps to format payload as json string so that later can extract from logs and rebuild the data easily
        logger.debug(
            f"[COLLECT] Payload prepared: <features>{json.dumps(payload)}</features>"
        )

        # Make the POST request to the external service
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.seq_retriever_model_server_url}/{endpoint}",
                    json=payload,
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )

            # Handle response
            if response.status_code == 200:
                logger.debug(
                    f"[COLLECT] Response from external service: <result>{json.dumps(response.json())}</result>"
                )
                result = response.json()
                return SeqRetrieverResponse(result=result)
            else:
                error_message = (
                    f"[DEBUG] External service returned an error: {response.text}"
                )
                logger.error(error_message)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_message,
                )

        except httpx.HTTPError as e:
            error_message = f"[DEBUG] Error connecting to external service: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message) from e

    async def search_items_by_title(
        self, query: str, limit: int
    ) -> SearchByTitleResponse:
        """
        Search for items by title using text matching in Qdrant.
        """
        logger.info(f"Searching for items with title containing: {query}")

        # Create a filter for partial text matching on title
        search_filter = Filter(
            must=[FieldCondition(key="title", match=MatchText(text=query))]
        )

        # Use scroll method which is better for pure filtering operations
        hits, _ = self.services.ann_index.scroll(
            collection_name=cfg.vectorstore.qdrant.collection_name,
            scroll_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

        # Format results
        results = []
        for hit in hits:
            payload = hit.payload
            results.append(
                RecommendationItem(
                    score=1.0,  # Default score for text matches
                    main_category=payload.get("main_category", ""),
                    title=payload.get("title", ""),
                    average_rating=payload.get("average_rating", 0.0),
                    rating_number=payload.get("rating_number", 0),
                    price=payload.get("price"),
                    subtitle=payload.get("subtitle"),
                    image_url=payload.get("image_url", ""),
                    parent_asin=payload.get("parent_asin", ""),
                )
            )

        logger.info(f"Found {len(results)} items matching title search: {query}")

        return SearchByTitleResponse(
            items=results,
            debug_info=[
                f"Title search query: {query}",
                f"Results count: {len(results)}",
            ],
        )
