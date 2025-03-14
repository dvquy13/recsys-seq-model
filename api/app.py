import json
import os
import sys
from typing import Any, Dict, Optional

import httpx
import numpy as np
import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from qdrant_client import QdrantClient

from src.cfg import ConfigLoader
from src.dto import RetrieveContext

from .logging_utils import RequestIDMiddleware
from .utils import debug_logging_decorator

cfg = ConfigLoader("./cfg/common.yaml")

app = FastAPI()
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | request_id: {extra[rec_id]} - {message}",
)


SEQ_RETRIEVER_MODEL_SERVER_URL = os.getenv(
    "SEQ_RETRIEVER_MODEL_SERVER_URL", "http://localhost:3000"
)
REDIS_HOST = cfg.redis.host
REDIS_PORT = cfg.redis.port
QDRANT_URL = os.getenv("QDRANT_URL", cfg.vectorstore.qdrant.url)

redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
)
redis_feature_prev_items_key_prefix = cfg.redis.keys.recent_key_prefix
redis_output_popular_key = cfg.redis.keys.popular_key

ann_index = QdrantClient(url=QDRANT_URL)


def get_recommendations_from_redis(
    redis_key: str, count: Optional[int]
) -> Dict[str, Any]:
    rec_data = redis_client.get(redis_key)
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


def get_user_prev_interactions(user_id: str) -> Dict[str, Any]:
    key = redis_feature_prev_items_key_prefix + user_id
    data = redis_client.get(key)
    if not data:
        error_message = f"[DEBUG] No recommendations found for key: {key}"
        logger.error(error_message)
        raise HTTPException(status_code=404, detail=error_message)
    return {"recent_interactions": data.split("__")}


@app.post("/recs/retrieve", summary="Retrieve the candidate for recommendations")
@debug_logging_decorator
async def retrieve(
    ctx: RetrieveContext,
    count: Optional[int] = Query(10, description="Number of items to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    if len(ctx.user_ids_raw) > 0:
        logger.info(f"Getting recent interactions for user: {ctx.user_ids_raw[0]}")
        user_id = ctx.user_ids_raw[0]
        user_prev_interactions = get_user_prev_interactions(user_id)[
            "recent_interactions"
        ]
        logger.info(f"[DEBUG] {user_prev_interactions=}")
        curr_item_seq = ctx.item_seq_raw[0]
        ctx.item_seq_raw = [user_prev_interactions + curr_item_seq]
        logger.info(f"[DEBUG] {ctx=}")

    if len(ctx.item_seq_raw[0]) == 0:
        logger.info("Empty RetrieveContext, fallback to popular recommendations")
        return await get_recommendations_popular(count=count)

    query_embedding_resp = await seq_retriever(ctx, endpoint="get_query_embeddings")
    query_embedding = np.array(query_embedding_resp["query_embedding"])
    logger.info(f"[DEBUG] {query_embedding.shape=}")

    hits = ann_index.search(
        collection_name=cfg.vectorstore.qdrant.collection_name,
        query_vector=query_embedding[0],
        limit=count,
    )
    recommendations = [
        {"score": hit.model_dump()["score"], **hit.payload} for hit in hits
    ]
    return {"recommendations": recommendations, "ctx": ctx.model_dump()}


@app.get("/recs/popular", summary="Get popular items as recommendations")
@debug_logging_decorator
async def get_recommendations_popular(
    count: Optional[int] = Query(10, description="Number of popular items to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    recommendations = get_recommendations_from_redis(redis_output_popular_key, count)
    return {"recommendations": recommendations}


# New endpoint to connect to external service
@app.post("/vendor/seq_retriever", summary="Call SeqRetriever model endpoint")
@debug_logging_decorator
async def seq_retriever(
    ctx: RetrieveContext,
    endpoint: str,
    debug: bool = Query(False, description="Enable debug logging"),
):
    user_ids = ctx.user_ids_raw
    item_seq = ctx.item_seq_raw
    candidate_items = ctx.candidate_items_raw

    logger.debug(
        f"Calling seq_rating_predicting with user_ids: {user_ids}, item_seq: {item_seq} and item_ids: {candidate_items}"
    )

    # Step 1: Prepare the payload for the external service
    payload = {"ctx": ctx.model_dump()}

    # Using json.dumps to format payload as json string so that later can extract from logs and rebuild the data easily
    logger.debug(
        f"[COLLECT] Payload prepared: <features>{json.dumps(payload)}</features>"
    )

    # Step 2: Make the POST request to the external service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SEQ_RETRIEVER_MODEL_SERVER_URL}/{endpoint}",
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

        # Step 3: Handle response
        if response.status_code == 200:
            logger.debug(
                f"[COLLECT] Response from external service: <result>{json.dumps(response.json())}</result>"
            )
            result = response.json()
            return result
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
