from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config.configuration import load_config
from src.constants import MODEL_DIR, TOKENIZER_DIR, STAGE_02
from src.utils import create_directories
from src import get_logger

logger = get_logger(__name__)
CONFIG = load_config()


@dataclass
class BaseModelArtifact:
    model_path: str
    tokenizer_path: str
    model_name: str
    num_labels: int


class PrepareBaseModel:
    def __init__(self):
        self.model_name     = CONFIG.model.name
        self.model_path     = str(MODEL_DIR)
        self.tokenizer_path = str(TOKENIZER_DIR)
        self.num_labels     = 2   # binary classification: <=50K / >50K

    # ── internals ────────────────────────────────────────────────────

    def _load_tokenizer(self):
        logger.info(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Tokenizer loaded ✅")
        return tokenizer

    def _load_model(self):
        logger.info(f"Loading model: {self.model_name}  (num_labels={self.num_labels})")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,   # allows reusing pretrained weights
        )
        logger.info(f"Model loaded ✅  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model

    def _save(self, model, tokenizer):
        create_directories([self.model_path, self.tokenizer_path])

        model.save_pretrained(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

        tokenizer.save_pretrained(self.tokenizer_path)
        logger.info(f"Tokenizer saved to {self.tokenizer_path}")

    # ── public ───────────────────────────────────────────────────────

    def run(self) -> BaseModelArtifact:
        logger.info("=" * 50)
        logger.info("Starting Prepare Base Model Component")

        try:
            tokenizer = self._load_tokenizer()
            model     = self._load_model()
            self._save(model, tokenizer)

            artifact = BaseModelArtifact(
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                model_name=self.model_name,
                num_labels=self.num_labels,
            )
            logger.info(f"Prepare Base Model complete: {artifact}")
            return artifact

        except OSError as e:
            logger.error(f"Could not load model from HuggingFace — check model name or network: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Model loading runtime error (check torch/CUDA): {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in PrepareBaseModel: {e}")
            raise