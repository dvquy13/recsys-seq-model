import os
import sys
from typing import List

import bentoml
import torch
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from mlflow import MlflowClient
from src.cfg import ConfigLoader
from src.dto import RetrieveContext

cfg = ConfigLoader("./cfg/common.yaml")

model_name = cfg.train.retriever.mlf_model_name

with bentoml.importing():
    root_dir = os.path.abspath(os.path.join(__file__, "../.."))
    sys.path.insert(0, root_dir)

load_dotenv()

model_cfg = {
    model_name: {
        "name": model_name,
        "deploy_alias": "champion",
        "model_uri": f"models:/{model_name}@champion",
    },
}

for name, cfg in model_cfg.items():
    bentoml.mlflow.import_model(
        name,
        model_uri=cfg["model_uri"],
        signatures={
            "predict": {"batchable": True},
        },
    )


class GetQueryEmbeddingInput(BaseModel):
    item_seq: List[str]


@bentoml.service(name="seq_retriever_service")
class SeqRetrieverService:
    model_name = model_name
    bento_model = bentoml.models.get(model_name)

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)
        self.inferer = self.model.unwrap_python_model()

        model_name = self.model_name
        deploy_alias = model_cfg.get(model_name).get("deploy_alias")

        mlf_client = MlflowClient()
        self.model_version = mlf_client.get_model_version_by_alias(
            model_name, deploy_alias
        ).version
        logger.info(
            f"Model Version for '{model_name}' with alias '{deploy_alias}': {self.model_version}"
        )

    def _augment_response(self, resp: dict, ctx: RetrieveContext) -> dict:
        """
        Helper method to DRY the common response augmentation.
        """
        resp["metadata"] = {
            "model_version": self.model_version,
            "model_name": self.model_name,
        }
        resp["ctx"] = ctx.model_dump()
        return resp

    @bentoml.api
    def predict(self, ctx: RetrieveContext):
        resp = self.model.predict(ctx.model_dump())
        return self._augment_response(resp, ctx)

    @bentoml.api
    def get_query_embeddings(self, ctx: RetrieveContext):
        item_seq = [
            self.inferer.idm.get_item_index(item_id) for item_id in ctx.item_seq_raw[0]
        ]
        inputs = {"item_seq": torch.tensor([item_seq])}
        query_embedding = self.inferer.model.get_query_embeddings(inputs)
        resp = {"query_embedding": query_embedding.detach().numpy().tolist()}
        return self._augment_response(resp, ctx)
