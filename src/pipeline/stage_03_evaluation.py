import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.components.evaluation import Evaluation, EvaluationArtifact
from src.components.data_ingestion import DataIngestionArtifact
from src.components.prepare_base_model import BaseModelArtifact
from src.constants import STAGE_03
from src import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    def __init__(
        self,
        data_artifact: DataIngestionArtifact,
        model_artifact: BaseModelArtifact,
    ):
        self.component = Evaluation(
            model_path=model_artifact.model_path,
            tokenizer_path=model_artifact.tokenizer_path,
            data_path=data_artifact.processed_data_path,
        )

    def run(self) -> EvaluationArtifact:
        logger.info(f">>>>>> {STAGE_03} started <<<<<<")
        try:
            artifact = self.component.run()
            logger.info(f">>>>>> {STAGE_03} completed <<<<<<\n\nx==========x")

            # surface key results clearly in the log
            logger.info(f"  Accuracy              : {artifact.overall_accuracy:.4f}")
            logger.info(f"  ROC-AUC               : {artifact.roc_auc:.4f}")
            logger.info(
                f"  Bias verdict          : "
                f"{'ALL CHECKS PASSED ✅' if artifact.all_bias_checks_passed else 'BIAS DETECTED ⚠️'}"
            )
            if artifact.flagged_attributes:
                logger.warning(
                    f"  Flagged attributes    : {artifact.flagged_attributes}"
                )

            return artifact

        except FileNotFoundError as e:
            logger.error(f"{STAGE_03} — missing file: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"{STAGE_03} — runtime error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{STAGE_03} — unexpected error: {e}")
            raise


if __name__ == "__main__":
    # quick standalone run using default paths from constants
    from src.components.data_ingestion import DataIngestionArtifact
    from src.components.prepare_base_model import BaseModelArtifact
    from src.constants import (
        RAW_DATA_FILE,
        PROCESSED_DATA_FILE,
        MODEL_DIR,
        TOKENIZER_DIR,
    )

    _data_artifact = DataIngestionArtifact(
        raw_data_path=str(RAW_DATA_FILE),
        processed_data_path=str(PROCESSED_DATA_FILE),
        row_count=0,
        column_count=0,
        dropped_rows=0,
        class_distribution={},
    )
    _model_artifact = BaseModelArtifact(
        model_path=str(MODEL_DIR),
        tokenizer_path=str(TOKENIZER_DIR),
        model_name="",
        num_labels=2,
    )

    pipeline = EvaluationPipeline(_data_artifact, _model_artifact)
    artifact = pipeline.run()

    print(f"\nMetrics saved to   : {artifact.metrics_path}")
    print(f"Bias report saved  : {artifact.bias_report_path}")