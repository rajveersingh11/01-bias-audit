import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.components.prepare_base_model import PrepareBaseModel, BaseModelArtifact
from src.constants import STAGE_02
from src import get_logger

logger = get_logger(__name__)


class PrepareBaseModelPipeline:
    def __init__(self):
        self.component = PrepareBaseModel()

    def run(self) -> BaseModelArtifact:
        logger.info(f">>>>>> {STAGE_02} started <<<<<<")
        try:
            artifact = self.component.run()
            logger.info(f">>>>>> {STAGE_02} completed <<<<<<\n\nx==========x")
            return artifact
        except OSError as e:
            logger.error(f"{STAGE_02} — model download failed: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"{STAGE_02} — runtime error (CUDA/torch): {e}")
            raise
        except Exception as e:
            logger.exception(f"{STAGE_02} — unexpected error: {e}")
            raise


if __name__ == "__main__":
    pipeline = PrepareBaseModelPipeline()
    artifact = pipeline.run()
    print(f"Model path     : {artifact.model_path}")
    print(f"Tokenizer path : {artifact.tokenizer_path}")
    print(f"Model name     : {artifact.model_name}")
    print(f"Num labels     : {artifact.num_labels}")