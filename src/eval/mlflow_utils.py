from IPython import get_ipython
from IPython.display import display
from loguru import logger

from mlflow.exceptions import MlflowException
from src.cfg import Config
from src.eval.compare_runs import ModelMetricsComparisonVisualizer
from src.sequence.trainer import LitSequenceRetriever


def is_running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # This typically indicates a Jupyter Notebook or JupyterLab environment
            return True
        elif shell == "TerminalInteractiveShell":
            # Running in a terminal with IPython
            return False
        else:
            # Other type (possibly a different environment)
            return False
    except NameError:
        # get_ipython() is not defined, so likely a standard Python interpreter
        return False


def check_register_new_champion(trainer: LitSequenceRetriever, cfg: Config):
    """
    Compare the new model run with the current champion model based on the ROC-AUC metric,
    and update the champion alias if the new model's performance meets or exceeds the required threshold.

    Parameters:
        trainer: An object containing the training process, with a logger that exposes the experiment (MLflow client).
        cfg: Configuration object with evaluation and training parameters (e.g., eval.min_roc_auc, train.retriever.mlf_model_name, author).

    Returns:
        None
    """
    # Get the MLflow client from the trainer's logger
    mlf_client = trainer.logger.experiment
    deploy_alias = "champion"
    curr_model_run_id = None
    min_roc_auc = cfg.eval.min_roc_auc

    # Attempt to retrieve the current champion model run ID via its alias
    try:
        curr_champion_model = mlf_client.get_model_version_by_alias(
            cfg.train.retriever.mlf_model_name, deploy_alias
        )
        curr_model_run_id = curr_champion_model.run_id
    except MlflowException as e:
        if "not found" in str(e).lower():
            logger.info(
                f"There is no {deploy_alias} alias for model {cfg.train.retriever.mlf_model_name}"
            )

    # Retrieve metrics from the new model run
    new_mlf_run = trainer.logger.experiment.get_run(trainer.logger.run_id)
    new_metrics = new_mlf_run.data.metrics
    roc_auc = new_metrics["roc_auc"]

    # If a current champion exists, compare its metrics with the new run
    if curr_model_run_id:
        curr_model_run_info = mlf_client.get_run(curr_model_run_id)
        curr_metrics = curr_model_run_info.data.metrics
        if (curr_roc_auc := curr_metrics["roc_auc"]) > min_roc_auc:
            logger.info(
                f"Current {deploy_alias} model has {curr_roc_auc:,.4f} ROC-AUC. Setting it to the deploy baseline..."
            )
            min_roc_auc = curr_roc_auc

        # Visualize and compare metrics between the new run and the current champion
        top_metrics = ["roc_auc"]
        vizer = ModelMetricsComparisonVisualizer(curr_metrics, new_metrics, top_metrics)
        logger.info("Comparing metrics between new run and current champion:")
        if is_running_in_notebook():
            display(vizer.compare_metrics_df())
            vizer.create_metrics_comparison_plot(n_cols=5)
            vizer.plot_diff()

    # Register the new champion if its ROC-AUC meets or exceeds the baseline
    if roc_auc < min_roc_auc:
        logger.info(
            f"Current run has ROC-AUC = {roc_auc:,.4f}, smaller than {min_roc_auc:,.4f}. Skip aliasing this model as the new {deploy_alias}."
        )
    else:
        logger.info("Aliasing the new model as champion...")
        # Get the most recent registered model version for the current run
        model_version = (
            mlf_client.get_registered_model(cfg.train.retriever.mlf_model_name)
            .latest_versions[0]
            .version
        )

        mlf_client.set_registered_model_alias(
            name=cfg.train.retriever.mlf_model_name,
            alias=deploy_alias,
            version=model_version,
        )

        mlf_client.set_model_version_tag(
            name=cfg.train.retriever.mlf_model_name,
            version=model_version,
            key="author",
            value=cfg.run.author,
        )
