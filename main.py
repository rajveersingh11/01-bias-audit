import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="torchao")
warnings.filterwarnings("ignore", category=UserWarning,   module="torchao")

import logging
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

import torch
torch.set_num_threads(torch.get_num_threads())

from src import get_logger
logger = get_logger(__name__)


# ── Stage 01 — Data Ingestion ─────────────────────────────────────────
STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
    data_artifact = DataIngestionPipeline().run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e


# ── Stage 02 — Prepare Base Model ────────────────────────────────────
STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
    model_artifact = PrepareBaseModelPipeline().run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e


# ── Stage 03 — Evaluation ────────────────────────────────────────────
STAGE_NAME = "Evaluation"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    from src.pipeline.stage_03_evaluation import EvaluationPipeline
    eval_artifact = EvaluationPipeline(
        data_artifact=data_artifact,
        model_artifact=model_artifact,
    ).run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e


# ── Stage 04 — Report Generation ─────────────────────────────────────
STAGE_NAME = "Report Generation"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    from src.pipeline.stage_04_report import ReportPipeline
    report_artifact = ReportPipeline(
        data_artifact=data_artifact,
        eval_artifact=eval_artifact,
    ).run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e


# ── Summary ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("PIPELINE COMPLETE")
logger.info("=" * 60)
logger.info(f"  Data      -> {data_artifact.processed_data_path}")
logger.info(f"  Model     -> {model_artifact.model_path}")
logger.info(f"  Metrics   -> {eval_artifact.metrics_path}")
logger.info(f"  Report    -> {report_artifact.report_path}")
logger.info(f"  Charts    -> {report_artifact.charts_dir}")
logger.info(f"  Accuracy  -> {eval_artifact.overall_accuracy:.4f}")
logger.info(f"  ROC-AUC   -> {eval_artifact.roc_auc:.4f}")
logger.info(
    f"  Result    -> "
    f"{'ALL BIAS CHECKS PASSED' if eval_artifact.all_bias_checks_passed else 'BIAS DETECTED'}"
)
if eval_artifact.flagged_attributes:
    logger.warning(f"  Flagged   -> {eval_artifact.flagged_attributes}")
logger.info("=" * 60)