import json
import os
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import HTTPException
from loguru import logger

from src.cfg import ConfigLoader
from src.dto import RetrieveContext

from .models import ItemsByIdsResponse, RecommendationResponse, SeqRetrieverResponse

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
        if len(ctx.user_ids_raw) > 0 and (user_id := ctx.user_ids_raw[0]):
            logger.info(f"Getting recent interactions for user: {user_id}")
            user_id = ctx.user_ids_raw[0]
            user_prev_interactions = self.get_user_prev_interactions(user_id)[
                "recent_interactions"
            ]
            logger.info(f"[DEBUG] {user_prev_interactions=}")
            curr_item_seq = ctx.item_seq_raw[0]
            ctx.item_seq_raw = [user_prev_interactions + curr_item_seq]
            logger.info(f"[DEBUG] {ctx=}")

        if len(ctx.item_seq_raw[0]) == 0:
            logger.info("Empty RetrieveContext, fallback to popular recommendations")
            return await self.get_popular_recommendations(count)

        query_embedding_resp = await self.call_seq_retriever(
            ctx, "get_query_embeddings"
        )
        query_embedding = np.array(query_embedding_resp.result["query_embedding"])
        logger.info(f"[DEBUG] {query_embedding.shape=}")

        hits = self.services.ann_index.search(
            collection_name=cfg.vectorstore.qdrant.collection_name,
            query_vector=query_embedding[0],
            limit=count,
        )
        recommendations = [
            {"score": hit.model_dump()["score"], **hit.payload} for hit in hits
        ]
        return RecommendationResponse(
            recommendations=recommendations,
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
            raise HTTPException(status_code=500, detail=error_message)
