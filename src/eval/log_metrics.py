import os
import warnings

import numpy as np
import pandas as pd
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (
    FBetaTopKMetric,
    NDCGKMetric,
    PersonalizationMetric,
    PrecisionTopKMetric,
    RecallTopKMetric,
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

import mlflow
from src.cfg import Config
from src.viz import color_scheme

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module=r"evidently.metrics.recsys.precision_recall_k",
)


def log_ranking_metrics(cfg: Config, eval_df):
    column_mapping = ColumnMapping(
        recommendations_type="rank",
        target=cfg.data.rating_col,
        prediction="rec_ranking",
        item_id=cfg.data.item_col,
        user_id=cfg.data.user_col,
    )

    report = Report(
        metrics=[
            NDCGKMetric(k=cfg.eval.top_k_rerank),
            RecallTopKMetric(k=cfg.eval.top_k_retrieve),
            PrecisionTopKMetric(k=cfg.eval.top_k_rerank),
            FBetaTopKMetric(k=cfg.eval.top_k_rerank),
            PersonalizationMetric(k=cfg.eval.top_k_rerank),
        ],
        options=[color_scheme],
    )

    report.run(reference_data=None, current_data=eval_df, column_mapping=column_mapping)

    evidently_report_fp = f"{cfg.run.run_persist_dir}/evidently_report.html"
    os.makedirs(cfg.run.run_persist_dir, exist_ok=True)
    report.save_html(evidently_report_fp)

    if cfg.run.log_to_mlflow:
        mlflow.log_artifact(evidently_report_fp)
        for metric_result in report.as_dict()["metrics"]:
            metric = metric_result["metric"]
            if metric == "PersonalizationMetric":
                metric_value = float(metric_result["result"]["current_value"])
                mlflow.log_metric(f"val_{metric}", metric_value)
                continue
            result = metric_result["result"]["current"].to_dict()
            for kth, metric_value in result.items():
                mlflow.log_metric(f"val_{metric}_at_k_as_step", metric_value, step=kth)

    return report


def log_classification_metrics(
    cfg: Config,
    eval_classification_df,
    target_col="label",
    prediction_col="classification_proba",
):
    column_mapping = ColumnMapping(target=target_col, prediction=prediction_col)
    classification_performance_report = Report(
        metrics=[
            ClassificationPreset(),
        ]
    )

    classification_performance_report.run(
        reference_data=None,
        current_data=eval_classification_df[[target_col, prediction_col]],
        column_mapping=column_mapping,
    )

    evidently_report_fp = (
        f"{cfg.run.run_persist_dir}/evidently_report_classification.html"
    )
    os.makedirs(cfg.run.run_persist_dir, exist_ok=True)
    classification_performance_report.save_html(evidently_report_fp)

    if cfg.run.log_to_mlflow:
        mlflow.log_artifact(evidently_report_fp)
        for metric_result in classification_performance_report.as_dict()["metrics"]:
            metric = metric_result["metric"]
            if metric == "ClassificationQualityMetric":
                roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                mlflow.log_metric("val_roc_auc", roc_auc)
                continue
            if metric == "ClassificationPRTable":
                columns = [
                    "top_perc",
                    "count",
                    "prob",
                    "tp",
                    "fp",
                    "precision",
                    "recall",
                ]
                table = metric_result["result"]["current"][1]
                table_df = pd.DataFrame(table, columns=columns)
                for i, row in table_df.iterrows():
                    prob = int(row["prob"] * 100)  # MLflow step only takes int
                    precision = float(row["precision"])
                    recall = float(row["recall"])
                    mlflow.log_metric(
                        "val_precision_at_prob_as_threshold_step", precision, step=prob
                    )
                    mlflow.log_metric(
                        "val_recall_at_prob_as_threshold_step", recall, step=prob
                    )
                break

    return classification_performance_report


def mse(predictions, ratings):
    predictions = np.array(predictions)
    ratings = np.array(ratings)
    return np.mean((predictions - ratings) ** 2)
