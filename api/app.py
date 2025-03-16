import os
import sys
from typing import Optional

import redis
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from qdrant_client import QdrantClient

from src.cfg import ConfigLoader
from src.dto import RetrieveContext
from src.id_mapper import IDMapper
from src.io_utils import init_s3_client

from .logging_utils import RequestIDMiddleware
from .models import (
    ItemsByIdsRequest,
    ItemsByIdsResponse,
    PopularItemsRequest,
    RecommendationResponse,
    SeqRetrieverRequest,
    SeqRetrieverResponse,
)
from .services import RecommendationService
from .utils import debug_logging_decorator

# Configuration and initialization
cfg = ConfigLoader("./cfg/common.yaml")


# Initialize ID mapper
def init_id_mapper():
    idm_fp = "./idm.json"
    if not os.path.exists(cfg.data.train_features_fp):
        s3 = init_s3_client()
        bucket_name = cfg.data.bucket_name
        idm_key = cfg.data.idm_fp.split("/")[-1]
        logger.info(f"Downloading {idm_key} from S3...")
        s3.download_file(bucket_name, idm_key, idm_fp)
    return IDMapper().load(idm_fp)


idm = init_id_mapper()

# Environment variables and constants
SEQ_RETRIEVER_MODEL_SERVER_URL = os.getenv(
    "SEQ_RETRIEVER_MODEL_SERVER_URL", "http://localhost:3000"
)
REDIS_HOST = cfg.redis.host
REDIS_PORT = cfg.redis.port
QDRANT_URL = os.getenv("QDRANT_URL", cfg.vectorstore.qdrant.url)

# Initialize clients
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
)
redis_feature_prev_items_key_prefix = cfg.redis.keys.recent_key_prefix
redis_output_popular_key = cfg.redis.keys.popular_key

ann_index = QdrantClient(url=QDRANT_URL)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | request_id: {extra[rec_id]} - {message}",
)


# Dependency for common services
class Services:
    def __init__(self):
        self.redis_client = redis_client
        self.ann_index = ann_index
        self.idm = idm


def get_services():
    return Services()


def get_recommendation_service(services: Services = Depends(get_services)):
    return RecommendationService(services)


# API Endpoints
@app.post(
    "/recs/retrieve",
    summary="Retrieve the candidate for recommendations",
    response_model=RecommendationResponse,
)
@debug_logging_decorator
async def retrieve(
    ctx: RetrieveContext,
    count: Optional[int] = 10,
    debug: bool = False,
    rec_service: RecommendationService = Depends(get_recommendation_service),
):
    return await rec_service.retrieve_recommendations(ctx, count)


@app.get(
    "/recs/popular",
    summary="Get popular items as recommendations",
    response_model=RecommendationResponse,
)
@debug_logging_decorator
async def get_recommendations_popular(
    request: PopularItemsRequest = Depends(),
    rec_service: RecommendationService = Depends(get_recommendation_service),
):
    return await rec_service.get_popular_recommendations(request.count)


@app.post(
    "/vendor/seq_retriever",
    summary="Call SeqRetriever model endpoint",
    response_model=SeqRetrieverResponse,
)
@debug_logging_decorator
async def seq_retriever(
    request: SeqRetrieverRequest,
    rec_service: RecommendationService = Depends(get_recommendation_service),
):
    return await rec_service.call_seq_retriever(request.ctx, request.endpoint)


@app.post(
    "/items/get_by_ids",
    summary="Retrieve items by their IDs",
    response_model=ItemsByIdsResponse,
)
@debug_logging_decorator
async def get_items_by_ids(
    request: ItemsByIdsRequest,
    rec_service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Retrieve items by their IDs. The IDs will be mapped to indices before querying the vector store.
    """
    return await rec_service.get_items_by_ids(request.item_ids)
