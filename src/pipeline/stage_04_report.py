import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.components.report import ReportGenerator, ReportArtifact
from src.components.evaluation import EvaluationArtifact
from src.components.data_ingestion import DataIngestionArtifact
from src.constants import TARGET_COLUMN
from src.utils import load_dataframe, load_json
from src import get_logger

logger = get_logger(__name__)

STAGE_NAME = "Stage 04 — Report Generation"


class ReportPipeline:
    def __init__(
        self,
        data_artifact: DataIngestionArtifact,
        eval_artifact: EvaluationArtifact,
    ):
        self.data_artifact = data_artifact
        self.eval_artifact = eval_artifact

    def run(self) -> ReportArtifact:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        try:
            # load processed data
            df = load_dataframe(self.data_artifact.processed_data_path)

            # load predictions from cached CSV written by evaluation stage
            import pandas as pd
            from pathlib import Path

            cache_files = list(Path("outputs/cache").glob("predictions_*.csv"))
            if cache_files:
                # use most recently written cache file
                cache_path = max(cache_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Loading predictions from cache: {cache_path}")
                preds_df         = pd.read_csv(cache_path)
                df["prediction"] = preds_df["prediction"].tolist()
                df["prob_gt50k"] = preds_df["prob_gt50k"].tolist()
            else:
                # fallback — majority class placeholder
                logger.warning(
                    "No prediction cache found — using majority class placeholder. "
                    "Run Evaluation stage first for accurate charts."
                )
                df["prediction"] = [0] * len(df)
                df["prob_gt50k"] = [0.25] * len(df)

            y_true = df[TARGET_COLUMN].tolist()
            y_pred = df["prediction"].tolist()
            y_prob = df["prob_gt50k"].tolist()

            generator = ReportGenerator(
                metrics_path=self.eval_artifact.metrics_path,
                bias_path=self.eval_artifact.bias_report_path,
            )
            artifact = generator.run(df, y_true, y_pred, y_prob)

            logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
            return artifact

        except Exception as e:
            logger.exception(f"{STAGE_NAME} failed: {e}")
            raise