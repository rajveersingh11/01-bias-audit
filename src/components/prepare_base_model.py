import gc
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.configuration import load_config
from src.constants import MODEL_DIR, TOKENIZER_DIR
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
        self.num_labels     = 2

    def _is_cached(self) -> bool:
        """Return True if model and tokenizer already saved to disk."""
        model_dir     = Path(self.model_path)
        tokenizer_dir = Path(self.tokenizer_path)
        model_saved     = model_dir.exists() and any(model_dir.iterdir())
        tokenizer_saved = tokenizer_dir.exists() and any(tokenizer_dir.iterdir())
        if model_saved and tokenizer_saved:
            logger.info(f"Model cache found — skipping download and save")
            return True
        return False

    def _load_tokenizer(self):
        logger.info(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Tokenizer loaded")
        return tokenizer

    def _load_model(self):
        logger.info(f"Loading model: {self.model_name}  (num_labels={self.num_labels})")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded — Parameters: {param_count:,}")
        return model

    def _save(self, model, tokenizer) -> None:
        """
        Save via temp dir then move — avoids Windows os error 1224.

        HuggingFace memory-maps the source .safetensors file when loading,
        so Windows locks it. Writing to a temp dir first and then replacing
        the target bypasses that lock entirely.
        """
        model_target     = Path(self.model_path)
        tokenizer_target = Path(self.tokenizer_path)

        # ── save model to temp dir, then replace target ───────────
        with tempfile.TemporaryDirectory() as tmp:
            tmp_model = Path(tmp) / "model"
            tmp_tok   = Path(tmp) / "tokenizer"
            tmp_model.mkdir()
            tmp_tok.mkdir()

            logger.info("Saving model to temp directory...")
            try:
                model.save_pretrained(str(tmp_model), safe_serialization=True)
                logger.info("Saved as safetensors")
            except Exception as e:
                logger.warning(f"safetensors failed ({e}) — retrying as .bin")
                # clear partial write
                shutil.rmtree(str(tmp_model))
                tmp_model.mkdir()
                model.save_pretrained(str(tmp_model), safe_serialization=False)
                logger.info("Saved as .bin")

            logger.info("Saving tokenizer to temp directory...")
            tokenizer.save_pretrained(str(tmp_tok))

            # release memory-map before replacing files
            del model
            del tokenizer
            gc.collect()

            # replace target directories atomically
            if model_target.exists():
                shutil.rmtree(str(model_target))
            shutil.copytree(str(tmp_model), str(model_target))
            logger.info(f"Model moved to {model_target}")

            if tokenizer_target.exists():
                shutil.rmtree(str(tokenizer_target))
            shutil.copytree(str(tmp_tok), str(tokenizer_target))
            logger.info(f"Tokenizer moved to {tokenizer_target}")

    def run(self) -> BaseModelArtifact:
        logger.info("=" * 50)
        logger.info("Starting Prepare Base Model Component")

        try:
            # skip everything if already cached
            if self._is_cached():
                artifact = BaseModelArtifact(
                    model_path=self.model_path,
                    tokenizer_path=self.tokenizer_path,
                    model_name=self.model_name,
                    num_labels=self.num_labels,
                )
                logger.info(f"Prepare Base Model complete (from cache): {artifact}")
                return artifact

            create_directories([self.model_path, self.tokenizer_path])
            tokenizer = self._load_tokenizer()
            model     = self._load_model()
            self._save(model, tokenizer)   # model/tokenizer deleted inside _save

            artifact = BaseModelArtifact(
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                model_name=self.model_name,
                num_labels=self.num_labels,
            )
            logger.info(f"Prepare Base Model complete: {artifact}")
            return artifact

        except OSError as e:
            logger.error(f"Could not load model — check model name or network: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error (check torch/CUDA): {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in PrepareBaseModel: {e}")
            raise


if __name__ == "__main__":
    component = PrepareBaseModel()
    artifact  = component.run()
    print(f"Model path     : {artifact.model_path}")
    print(f"Tokenizer path : {artifact.tokenizer_path}")