import os
import warnings

import lightning as L
import pandas as pd
import torch
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
        self.evaluate_ranking = evaluate_ranking
        self.idm = idm
        self.args = args
        self.accelerator = accelerator
        self.checkpoint_callback = checkpoint_callback

    def training_step(self, batch, batch_idx):
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]
        input_item_sequences = batch["item_sequence"]

        labels = batch["rating"].float()
        predictions = self.model.forward(
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
        predictions = self.model.forward(
            input_user_ids, input_item_sequences, input_item_ids
        ).view(labels.shape)

        loss_fn = self._get_loss_fn()
        loss = loss_fn(predictions, labels)

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )

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

    def on_fit_end(self):
        if self.checkpoint_callback:
            logger.info(
                f"Loading best model from {self.checkpoint_callback.best_model_path}..."
            )
            self.model = LitSequenceRatingPrediction.load_from_checkpoint(
                self.checkpoint_callback.best_model_path, model=self.model
            ).model
        self.model = self.model.to(self._get_device())
        if self.evaluate_ranking:
            logger.info("Logging ranking metrics...")
            self._log_ranking_metrics()

    def _log_ranking_metrics(self):
        self.model.eval()

        timestamp_col = self.args.timestamp_col
        rating_col = self.args.rating_col
        user_col = self.args.user_col
        item_col = self.args.item_col
        top_K = self.args.top_K
        top_k = self.args.top_k
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
            k=top_K,
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
            k=top_K,
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
                NDCGKMetric(k=top_k),
                RecallTopKMetric(k=top_K),
                PrecisionTopKMetric(k=top_k),
                FBetaTopKMetric(k=top_k),
                PersonalizationMetric(k=top_k),
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
        return nn.MSELoss()

    def _get_device(self):
        return self.accelerator
