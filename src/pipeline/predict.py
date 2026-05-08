import sys
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="torchao")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import pandas as pd
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.configuration import load_config
from src.constants import MODEL_DIR, TOKENIZER_DIR
from src import get_logger

logger = get_logger(__name__)
CONFIG = load_config()


# ── prediction result ─────────────────────────────────────────────────

@dataclass
class PredictionResult:
    input_data:      dict
    prediction:      int
    label:           str
    confidence:      float
    prob_leq_50k:    float
    prob_gt_50k:     float
    flagged_fields:  list[str] = field(default_factory=list)


# ── pipeline ──────────────────────────────────────────────────────────

class PredictionPipeline:
    """
    Encapsulates inference logic for a single input record.
    Designed for integration with Flask or any web service framework.

    Usage:
        pipeline = PredictionPipeline()
        result   = pipeline.predict({
            "age": 39,
            "education": "Bachelors",
            "occupation": "Exec-managerial",
            "hours.per.week": 40,
            "sex": "Male",
            "race": "White"
        })
    """

    # required fields — validated before inference
    REQUIRED_FIELDS = [
        "age",
        "education",
        "occupation",
        "hours.per.week",
        "sex",
        "race",
    ]

    LABEL_MAP = {0: "<=50K", 1: ">50K"}

    def __init__(
        self,
        model_path:     str = str(MODEL_DIR),
        tokenizer_path: str = str(TOKENIZER_DIR),
    ):
        self.model_path     = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device         = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model     = None
        self._tokenizer = None

        logger.info(
            f"PredictionPipeline initialised — device: {self.device} | "
            f"model: {self.model_path}"
        )

    # ── lazy loading ──────────────────────────────────────────────────

    def _load(self) -> None:
        """
        Lazy-load model and tokenizer on first predict() call.
        Avoids loading 270MB into memory at import time when
        the module is imported by Flask at startup.
        """
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                "  -> Run main.py to train and save the model first"
            )
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {self.tokenizer_path}\n"
                "  -> Run main.py to train and save the model first"
            )

        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))

        logger.info(f"Loading model from {self.model_path}")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )

        # int8 quantization on CPU for faster inference
        if self.device.type == "cpu":
            import torch.quantization
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = torch.quantization.quantize_dynamic(
                    self._model, {torch.nn.Linear}, dtype=torch.qint8
                )

        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("Model ready")

    # ── validation ────────────────────────────────────────────────────

    def _validate(self, input_data: dict) -> list[str]:
        """
        Check for missing required fields.
        Returns list of missing field names.
        """
        return [f for f in self.REQUIRED_FIELDS if f not in input_data
                or input_data[f] is None or str(input_data[f]).strip() == ""]

    def _sanitize(self, input_data: dict) -> dict:
        """Strip whitespace from string values."""
        return {
            k: str(v).strip() if isinstance(v, str) else v
            for k, v in input_data.items()
        }

    # ── text construction ─────────────────────────────────────────────

    def _build_text(self, data: dict) -> str:
        """
        Convert input dict to natural language string.
        Must mirror exactly what evaluation.py uses — keep in sync.
        """
        return (
            f"age {data.get('age')}, "
            f"education {data.get('education')}, "
            f"occupation {data.get('occupation')}, "
            f"hours {data.get('hours.per.week')}, "
            f"sex {data.get('sex')}, "
            f"race {data.get('race')}"
        )

    # ── inference ─────────────────────────────────────────────────────

    def _infer(self, text: str) -> tuple[int, float, float]:
        """
        Run model inference on a single text string.
        Returns (prediction, prob_leq_50k, prob_gt_50k).
        """
        enc = self._tokenizer(
            text,
            max_length=CONFIG.model.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self._model(**enc).logits
            probs  = F.softmax(logits, dim=-1).squeeze()

        prob_leq = round(float(probs[0]), 4)
        prob_gt  = round(float(probs[1]), 4)
        pred     = int(torch.argmax(probs).item())

        return pred, prob_leq, prob_gt

    # ── public API ────────────────────────────────────────────────────

    def predict(self, input_data: dict) -> PredictionResult:
        """
        Run inference on a single input record.

        Args:
            input_data: dict with keys matching REQUIRED_FIELDS

        Returns:
            PredictionResult dataclass

        Raises:
            ValueError:      if required fields are missing
            FileNotFoundError: if model/tokenizer not found on disk
        """
        # lazy load
        self._load()

        # validate
        missing = self._validate(input_data)
        if missing:
            raise ValueError(
                f"Missing required fields: {missing}\n"
                f"Required: {self.REQUIRED_FIELDS}"
            )

        # sanitize
        data = self._sanitize(input_data)

        # build text
        text = self._build_text(data)
        logger.debug(f"Input text: {text}")

        # infer
        pred, prob_leq, prob_gt = self._infer(text)
        label      = self.LABEL_MAP[pred]
        confidence = prob_gt if pred == 1 else prob_leq

        result = PredictionResult(
            input_data=data,
            prediction=pred,
            label=label,
            confidence=confidence,
            prob_leq_50k=prob_leq,
            prob_gt_50k=prob_gt,
        )

        logger.info(
            f"Prediction: {label}  "
            f"(confidence: {confidence:.4f}  |  "
            f"P(<=50K)={prob_leq:.4f}  P(>50K)={prob_gt:.4f})"
        )
        return result

    def predict_batch(self, records: list[dict]) -> list[PredictionResult]:
        """
        Run inference on a list of input records.
        Validates all records before running any inference.

        Args:
            records: list of input dicts

        Returns:
            list of PredictionResult
        """
        self._load()

        if not records:
            raise ValueError("records list is empty")

        # validate all records up front
        errors = {}
        for i, record in enumerate(records):
            missing = self._validate(record)
            if missing:
                errors[i] = missing
        if errors:
            raise ValueError(f"Validation failed for records at indices: {errors}")

        results = []
        for record in records:
            results.append(self.predict(record))

        logger.info(f"Batch prediction complete — {len(results)} records")
        return results

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a pandas DataFrame.
        Adds 'prediction', 'label', 'prob_leq_50k', 'prob_gt_50k' columns.

        Args:
            df: DataFrame with columns matching REQUIRED_FIELDS

        Returns:
            DataFrame with prediction columns added
        """
        self._load()

        records = df.to_dict(orient="records")
        results = self.predict_batch(records)

        df = df.copy()
        df["prediction"]  = [r.prediction  for r in results]
        df["label"]       = [r.label        for r in results]
        df["confidence"]  = [r.confidence   for r in results]
        df["prob_leq_50k"]= [r.prob_leq_50k for r in results]
        df["prob_gt_50k"] = [r.prob_gt_50k  for r in results]

        return df

    def health_check(self) -> dict:
        """
        Returns model status — used by Flask /health endpoint.
        """
        try:
            self._load()
            test_result = self.predict({
                "age": 30,
                "education": "Bachelors",
                "occupation": "Prof-specialty",
                "hours.per.week": 40,
                "sex": "Male",
                "race": "White",
            })
            return {
                "status":      "healthy",
                "model_path":  str(self.model_path),
                "device":      str(self.device),
                "test_pred":   test_result.label,
                "test_conf":   test_result.confidence,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


# ── Flask integration example ─────────────────────────────────────────

"""
# app.py — drop-in Flask service

from flask import Flask, request, jsonify
from predict import PredictionPipeline

app      = Flask(__name__)
pipeline = PredictionPipeline()   # loaded once at startup


@app.route("/health", methods=["GET"])
def health():
    return jsonify(pipeline.health_check())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        result = pipeline.predict(data)
        return jsonify({
            "prediction":   result.prediction,
            "label":        result.label,
            "confidence":   result.confidence,
            "prob_leq_50k": result.prob_leq_50k,
            "prob_gt_50k":  result.prob_gt_50k,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        records = request.get_json(force=True)
        if not isinstance(records, list):
            return jsonify({"error": "Expected a JSON array of records"}), 400
        results = pipeline.predict_batch(records)
        return jsonify([{
            "prediction":   r.prediction,
            "label":        r.label,
            "confidence":   r.confidence,
            "prob_leq_50k": r.prob_leq_50k,
            "prob_gt_50k":  r.prob_gt_50k,
        } for r in results])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
"""


# ── standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = PredictionPipeline()

    # single prediction
    print("\n── Single prediction ────────────────────────────────")
    result = pipeline.predict({
        "age":           39,
        "education":     "Bachelors",
        "occupation":    "Exec-managerial",
        "hours.per.week": 40,
        "sex":           "Male",
        "race":          "White",
    })
    print(f"  Prediction   : {result.label}")
    print(f"  Confidence   : {result.confidence:.4f}")
    print(f"  P(<=50K)     : {result.prob_leq_50k:.4f}")
    print(f"  P(>50K)      : {result.prob_gt_50k:.4f}")

    # batch prediction
    print("\n── Batch prediction ─────────────────────────────────")
    batch = [
        {"age": 28, "education": "Masters",   "occupation": "Prof-specialty",
         "hours.per.week": 60, "sex": "Female", "race": "White"},
        {"age": 52, "education": "HS-grad",   "occupation": "Craft-repair",
         "hours.per.week": 45, "sex": "Male",   "race": "Black"},
        {"age": 35, "education": "Bachelors", "occupation": "Sales",
         "hours.per.week": 50, "sex": "Male",   "race": "Asian-Pac-Islander"},
    ]
    results = pipeline.predict_batch(batch)
    for i, r in enumerate(results):
        print(f"  [{i}] {r.label:<8}  conf={r.confidence:.4f}  "
              f"P(>50K)={r.prob_gt_50k:.4f}")

    # health check
    print("\n── Health check ─────────────────────────────────────")
    health = pipeline.health_check()
    for k, v in health.items():
        print(f"  {k:<15}: {v}")