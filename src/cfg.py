import json
import os
from typing import Literal, Optional

import torch
import yaml
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
from pydantic import BaseModel

import mlflow


def substitute_env_variables(obj):
    """
    Recursively substitute environment variables in strings within a dict or list.
    """
    if isinstance(obj, dict):
        return {key: substitute_env_variables(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_variables(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


class DataConfig(BaseModel):
    class HFDatasets(BaseModel):
        name: str
        mcauley_variant: str

    hf_datasets: HFDatasets

    train_fp: str
    val_fp: str
    idm_fp: str
    metadata_fp: str
    train_features_fp: str
    val_features_fp: str
    full_features_neg_fp: str
    train_features_neg_fp: str
    val_features_neg_fp: str

    user_col: str
    item_col: str
    rating_col: str
    timestamp_col: str


class TrainConfig(BaseModel):
    label_format: Literal["binary", "rating"]
    learning_rate: float = 0.01
    batch_size: int = 32
    max_epochs: int = 10
    early_stopping_patience: int = 5
    device: Optional[str] = None

    class RetrieverConfig(BaseModel):
        model_classname: str = "SequenceRetriever"
        mlf_model_name: str = "sequence_retriever"

    class SequenceConfig(BaseModel):
        sequence_length: int = 10

    embedding_dim: int = 128
    dropout: Optional[float] = None
    l2_reg: Optional[float] = None

    retriever: RetrieverConfig
    sequence: SequenceConfig


class RunConfig(BaseModel):
    author: str = ""
    testing: bool = False
    log_to_mlflow: bool = False
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    run_persist_dir: Optional[str] = None
    random_seed: int = 41


class EvalConfig(BaseModel):
    top_k_retrieve: int = 100
    top_k_rerank: int = 10
    min_roc_auc: float = 0.5


class QdrantConfig(BaseModel):
    classname: str = "QdrantVectorStore"
    url: Optional[str] = None
    collection_name: Optional[str] = None


class VectorstoreConfig(BaseModel):
    qdrant: QdrantConfig


class SampleConfig(BaseModel):
    sample_users: int = 10000
    min_val_records: int = 5000
    min_user_interactions: int = 5
    min_item_interactions: int = 10


class Config(BaseModel):
    run: RunConfig
    data: DataConfig
    sample: SampleConfig
    train: TrainConfig
    eval: EvalConfig
    vectorstore: VectorstoreConfig


def deep_update(original: dict, updates: dict) -> dict:
    """
    Recursively update a dictionary with values from another dictionary.
    If the value is a dict in both, it performs a deep merge.
    """
    for key, value in updates.items():
        if (
            key in original
            and isinstance(original[key], dict)
            and isinstance(value, dict)
        ):
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flatten a nested dictionary. Lists are converted to JSON strings.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v)
        else:
            items[new_key] = v
    return items


class ConfigLoader:
    def __init__(self, filepath: str = "config.yaml"):
        self.filepath = filepath
        self.config = self.get_config(filepath)

    @staticmethod
    def load_yaml_config(filepath: str) -> dict:
        """Load the YAML config file into a dictionary and substitute env variables."""
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        return substitute_env_variables(config_dict)

    @classmethod
    def get_config(cls, filepath: str = "config.yaml") -> Config:
        """
        Load and validate the shared configuration using Pydantic.
        IDEs will provide autocomplete for nested attributes, for example:
            config.data.train_fp
            config.train.learning_rate
        """
        config_dict = cls.load_yaml_config(filepath)
        return Config(**config_dict)

    def update_config(self, updates: dict) -> None:
        """
        Update the configuration with new values.
        This implementation uses a recursive merge to update nested keys.
        """
        updated_dict = deep_update(self.config.model_dump(), updates)
        self.config = Config(**updated_dict)

    def log_config_to_mlflow(self) -> None:
        """
        Log all configuration parameters to MLflow.
        Uses a dot-separated key for nested parameters (e.g., "train.learning_rate").
        """
        flat_config = flatten_dict(self.config.model_dump())
        for key, value in flat_config.items():
            mlflow.log_param(key, value)

        mlflow.log_artifact(self.filepath)

    def init(self):
        """
        Additional initialization.
        For example, if a run name is provided, set the run_persist_dir
        to an absolute path under the 'data' directory.
        """
        if self.run.run_name:
            self.run.run_persist_dir = os.path.abspath(
                f"data/{self.config.run.run_name}"
            )

        if not (mlflow_uri := os.environ.get("MLFLOW_TRACKING_URI")):
            logger.warning(
                "Environment variable MLFLOW_TRACKING_URI is not set. Setting self.run.log_to_mlflow to false."
            )
            self.run.log_to_mlflow = False

        if self.run.log_to_mlflow:
            logger.info(
                f"Setting up MLflow experiment {self.run.experiment_name} - run {self.run.run_name}..."
            )
            mlflow.set_experiment(self.run.experiment_name)
            mlflow.start_run(run_name=self.run.run_name)
            self._mlf_logger = MLFlowLogger(
                experiment_name=self.run.experiment_name,
                run_id=mlflow.active_run().info.run_id,  # If use run_name without run_id the it would create a new run!
                tracking_uri=mlflow_uri,
                log_model=True,
            )

        if self.train.device is None:
            self.train.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        return self

    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying config.
        This allows you to access config values directly, e.g., cfg.data instead of cfg.config.data.
        However, when calling something like `self._mlf_logger` Python still retrieves the attribute from `self` first before delegating to `getattr`.
        """
        return getattr(self.config, attr)

    def __repr__(self):
        return json.dumps(self.config.model_dump(), indent=2)
