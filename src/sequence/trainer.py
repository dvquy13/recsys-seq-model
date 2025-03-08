import os
import warnings

import lightning as L
import pandas as pd
import torch
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
from loguru import logger
from pydantic import BaseModel
from torch import nn
from torchmetrics import AUROC, AveragePrecision

from src.eval.utils import create_label_df, create_rec_df, merge_recs_with_target
from src.id_mapper import IDMapper
from src.viz import color_scheme

from .model import SequenceRatingPrediction

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module=r"evidently.metrics.recsys.precision_recall_k",
)


class LitSequenceRatingPrediction(L.LightningModule):
    def __init__(
        self,
        model: SequenceRatingPrediction,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
        evaluate_ranking: bool = False,
        idm: IDMapper = None,
        args: BaseModel = None,
        checkpoint_callback=None,
        accelerator: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        # Currently _log_ranking_metrics method has a lot of dependencies
        # It requires IDMapper and a bunch of other parameters
        # TODO: Refactor
        self.evaluate_ranking = evaluate_ranking
        self.idm = idm
        self.args = args
        self.accelerator = accelerator
        self.checkpoint_callback = checkpoint_callback

        # Initialize AUROC for binary classification
        self.val_roc_auc_metric = AUROC(task="binary")
        # Initialize PR-AUC for binary classification
        self.val_pr_auc_metric = AveragePrecision(task="binary")

    def training_step(self, batch, batch_idx):
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]
        input_item_sequences = batch["item_sequence"]

        # Ensure shapes match by squeezing if necessary
        labels = batch["rating"].float()
        predictions = self.model.predict(
            input_user_ids, input_item_sequences, input_item_ids
        ).view(labels.shape)

        loss_fn = self._get_loss_fn()
        loss = loss_fn(predictions, labels)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]
        input_item_sequences = batch["item_sequence"]

        labels = batch["rating"]
        predictions = self.model.predict(
            input_user_ids, input_item_sequences, input_item_ids
        ).view(labels.shape)

        loss_fn = self._get_loss_fn()
        loss = loss_fn(predictions, labels)

        # Update AUROC with current batch predictions and labels
        self.val_roc_auc_metric.update(predictions, labels.int())
        # Update PR-AUC with current batch predictions and labels
        self.val_pr_auc_metric.update(predictions, labels.int())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )

        # Create the scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=2
            ),
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()

        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)

        # Compute and log the final ROC-AUC for the epoch
        roc_auc = self.val_roc_auc_metric.compute()
        pr_auc = self.val_pr_auc_metric.compute()

        self.log(
            "val_roc_auc",
            roc_auc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_pr_auc",
            pr_auc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Reset the metric for the next epoch
        self.val_roc_auc_metric.reset()
        self.val_pr_auc_metric.reset()

    def on_train_epoch_end(self):
        self.log_weight_norms(stage="train")

    def log_weight_norms(self, stage="train"):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.log(
                    f"{stage}_weight_norm_{name}",
                    param.norm().item(),
                    on_step=False,
                    logger=True,
                )

    def on_fit_end(self):
        if self.checkpoint_callback:
            logger.info(
                f"Loading best model from {self.checkpoint_callback.best_model_path}..."
            )
            self.model = LitSequenceRatingPrediction.load_from_checkpoint(
                self.checkpoint_callback.best_model_path, model=self.model
            ).model
        self.model = self.model.to(self._get_device())
        logger.info("Logging classification metrics...")
        self._log_classification_metrics()
        if self.evaluate_ranking:
            logger.info("Logging ranking metrics...")
            self._log_ranking_metrics()

    def _log_classification_metrics(self):
        # Set model to evaluation mode to disable dropout and batchnorm
        self.model.eval()

        val_loader = self.trainer.val_dataloaders

        labels = []
        classifications = []

        for _, batch_input in enumerate(val_loader):
            _input_user_ids = batch_input["user"].to(self._get_device())
            _input_item_ids = batch_input["item"].to(self._get_device())
            _input_item_sequences = batch_input["item_sequence"].to(self._get_device())
            _labels = batch_input["rating"].to(self._get_device())
            _classifications = self.model.predict(
                _input_user_ids, _input_item_sequences, _input_item_ids
            ).view(_labels.shape)

            labels.extend(_labels.cpu().detach().numpy())
            classifications.extend(_classifications.cpu().detach().numpy())

        eval_classification_df = pd.DataFrame(
            {
                "labels": labels,
                "classification_proba": classifications,
            }
        ).assign(label=lambda df: df["labels"].gt(0).astype(int))

        self.eval_classification_df = eval_classification_df

        # Evidently report
        target_col = "label"
        prediction_col = "classification_proba"
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

        evidently_report_fp = f"{self.log_dir}/evidently_report_classification.html"
        os.makedirs(self.log_dir, exist_ok=True)
        classification_performance_report.save_html(evidently_report_fp)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            mlf_client = self.logger.experiment
            mlf_client.log_artifact(run_id, evidently_report_fp)

            # Calculate PR-AUC using torchmetrics for MLflow
            labels_tensor = torch.tensor(
                eval_classification_df[target_col].values, dtype=torch.int
            )
            probs_tensor = torch.tensor(
                eval_classification_df["classification_proba"].values, dtype=torch.float
            )
            pr_auc_metric = AveragePrecision(task="binary")
            pr_auc = pr_auc_metric(probs_tensor, labels_tensor).item()
            mlf_client.log_metric(run_id, "pr_auc", pr_auc)

            for metric_result in classification_performance_report.as_dict()["metrics"]:
                metric = metric_result["metric"]
                if metric == "ClassificationQualityMetric":
                    roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                    mlf_client.log_metric(run_id, "roc_auc", roc_auc)
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
                        mlf_client.log_metric(
                            run_id,
                            "val_precision_at_prob_as_threshold_step",
                            precision,
                            step=prob,
                        )
                        mlf_client.log_metric(
                            run_id,
                            "val_recall_at_prob_as_threshold_step",
                            recall,
                            step=prob,
                        )
                    break

    def _log_ranking_metrics(self):
        self.model.eval()

        timestamp_col = self.args.timestamp_col
        rating_col = self.args.rating_col
        user_col = self.args.user_col
        item_col = self.args.item_col
        top_k_retrieve = self.args.top_k_retrieve
        top_k_rerank = self.args.top_k_rerank
        idm = self.idm

        val_df = self.trainer.val_dataloaders.dataset.df

        # Prepare recommendations using the last interaction per user
        to_rec_df = val_df.sort_values(timestamp_col, ascending=True).drop_duplicates(
            subset=[user_col]
        )
        recommendations = self.model.recommend(
            torch.tensor(to_rec_df["user_indice"].values, device=self._get_device()),
            torch.tensor(
                to_rec_df["item_sequence"].values.tolist(), device=self._get_device()
            ),
            k=top_k_retrieve,
            batch_size=4,
        )

        recommendations_df = pd.DataFrame(recommendations).pipe(
            create_rec_df, idm, user_col, item_col
        )

        label_df = create_label_df(
            val_df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            timestamp_col=timestamp_col,
        )

        eval_df = merge_recs_with_target(
            recommendations_df,
            label_df,
            k=top_k_retrieve,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

        self.eval_ranking_df = eval_df

        column_mapping = ColumnMapping(
            recommendations_type="rank",
            target=rating_col,
            prediction="rec_ranking",
            item_id=item_col,
            user_id=user_col,
        )

        report = Report(
            metrics=[
                NDCGKMetric(k=top_k_rerank),
                RecallTopKMetric(k=top_k_retrieve),
                PrecisionTopKMetric(k=top_k_rerank),
                FBetaTopKMetric(k=top_k_rerank),
                PersonalizationMetric(k=top_k_rerank),
            ],
            options=[color_scheme],
        )

        report.run(
            reference_data=None, current_data=eval_df, column_mapping=column_mapping
        )

        evidently_report_fp = f"{self.log_dir}/evidently_report_ranking.html"
        os.makedirs(self.log_dir, exist_ok=True)
        report.save_html(evidently_report_fp)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            mlf_client = self.logger.experiment
            mlf_client.log_artifact(run_id, evidently_report_fp)
            for metric_result in report.as_dict()["metrics"]:
                metric = metric_result["metric"]
                if metric == "PersonalizationMetric":
                    metric_value = float(metric_result["result"]["current_value"])
                    mlf_client.log_metric(run_id, f"val_{metric}", metric_value)
                    continue
                result = metric_result["result"]["current"].to_dict()
                for kth, metric_value in result.items():
                    mlf_client.log_metric(
                        run_id, f"val_{metric}_at_k_as_step", metric_value, step=kth
                    )

    def _get_loss_fn(self):
        return nn.BCELoss()

    def _get_device(self):
        return self.accelerator
