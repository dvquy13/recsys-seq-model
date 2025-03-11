import os
import sys

import bentoml
from dotenv import load_dotenv
from loguru import logger

from mlflow import MlflowClient
from src.cfg import ConfigLoader

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


@bentoml.service(name="seq_retriever_service")
class SeqRetrieverService:
    model_name = model_name
    bento_model = bentoml.models.get(model_name)

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

        model_name = self.model_name
        deploy_alias = model_cfg.get(model_name).get("deploy_alias")

        mlf_client = MlflowClient()
        self.model_version = mlf_client.get_model_version_by_alias(
            model_name, deploy_alias
        ).version
        logger.info(
            f"Model Version for '{model_name}' with alias '{deploy_alias}': {self.model_version}"
        )

    @bentoml.api
    def predict(self, input_data):
        rv = self.model.predict(input_data)
        rv["metadata"] = {
            "model_version": self.model_version,
            "model_name": self.model_name,
        }
        return rv
