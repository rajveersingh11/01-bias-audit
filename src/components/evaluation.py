import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path (allows running this script directly)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import torch.quantization
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from config.configuration import load_config
from src.constants import (
    MODEL_DIR,
    TOKENIZER_DIR,
    METRICS_DIR,
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    PROTECTED_ATTRIBUTES,
    STAGE_03,
)
from src.utils import (
    create_directories,
    save_json,
    load_dataframe,
    is_within_threshold,
)
from src import get_logger

logger = get_logger(__name__)
CONFIG = load_config()


# ── dataset wrapper ───────────────────────────────────────────────────

class InferenceDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ── artifacts ─────────────────────────────────────────────────────────

@dataclass
class EvaluationArtifact:
    metrics_path: str
    bias_report_path: str
    overall_accuracy: float
    roc_auc: float
    all_bias_checks_passed: bool
    flagged_attributes: list[str] = field(default_factory=list)


# ── component ─────────────────────────────────────────────────────────

class Evaluation:
    def __init__(
        self,
        model_path: str = str(MODEL_DIR),
        tokenizer_path: str = str(TOKENIZER_DIR),
        data_path: str = str(PROCESSED_DATA_FILE),
    ):
        self.model_path     = model_path
        self.tokenizer_path = tokenizer_path
        self.data_path      = data_path
        self.device         = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.metrics_path     = str(METRICS_DIR / "metrics.json")
        self.bias_report_path = str(METRICS_DIR / "bias_report.json")

    # ── model loading ─────────────────────────────────────────────────

    def _load_model_and_tokenizer(self):
        import warnings
        import platform

        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        logger.info(f"Loading model from {self.model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # ── optimization 3: quantization ─────────────────────────────────
        if self.device.type == "cpu":
            try:
                # torchao is the modern replacement (PyTorch 2.6+)
                import torchao
                from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
                quantize_(model, int8_dynamic_activation_int8_weight())
                logger.info("Model quantized to int8 via torchao")
            except ImportError:
                # fall back to legacy API — suppress the deprecation warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    model = torch.quantization.quantize_dynamic(
                        model,
                        {torch.nn.Linear},
                        dtype=torch.qint8,
                    )
                logger.info("Model quantized to int8 via torch.quantization (legacy)")

        model = model.to(self.device)
        model.eval()

        # ── optimization 5: torch.compile ────────────────────────────────
        # Inductor backend (default) needs cl.exe on Windows — not always available.
        # Use backend="eager" as a safe fallback that works everywhere.
        if hasattr(torch, "compile"):
            if platform.system() == "Windows":
                try:
                    model = torch.compile(model, backend="eager")
                    logger.info("Model compiled with torch.compile (eager backend)")
                except Exception as e:
                    logger.warning(f"torch.compile skipped on Windows — {e}")
            else:
                try:
                    model = torch.compile(model)
                    logger.info("Model compiled with torch.compile (inductor backend)")
                except Exception as e:
                    logger.warning(f"torch.compile skipped — {e}")

        logger.info(f"Model ready on {self.device}")
        return model, tokenizer

    # ── text construction ─────────────────────────────────────────────

    def _build_texts(self, df: pd.DataFrame) -> list[str]:
        """
        Convert each row into a natural language string for the model.
        Mirrors exactly what prepare_base_model.ipynb used — keep in sync.
        """
        return df.apply(
            lambda r: (
                f"age {r['age']}, education {r['education']}, "
                f"occupation {r['occupation']}, "
                f"hours {r['hours.per.week']}, "
                f"sex {r['sex']}, race {r['race']}"
            ),
            axis=1,
        ).tolist()

    # ── inference ─────────────────────────────────────────────────────

    def _run_inference(
        self, model, tokenizer, texts: list[str]
    ) -> tuple[list[int], list[float]]:
        """
        Batch inference over all texts with optimizations:
        - torch.inference_mode (faster than no_grad)
        - DataLoader with batch_size optimization
        """
        dataset    = InferenceDataset(texts, tokenizer, CONFIG.model.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG.model.batch_size,   # optimization 1: larger batch
            shuffle=False,
            num_workers=0,                         # 0 on Windows, 2+ on Linux/Mac
            pin_memory=self.device.type == "cuda", # optimization 4: pin_memory
        )

        all_preds, all_probs = [], []
        total = len(dataloader)

        # optimization 2: inference_mode (faster than no_grad)
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                batch   = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                probs   = F.softmax(outputs.logits, dim=-1)
                preds   = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs[:, 1].cpu().tolist())

                if (i + 1) % 20 == 0:
                    logger.debug(f"Inference progress: {i+1}/{total} batches ({(i+1)/total*100:.0f}%)")

        logger.info(f"Inference complete — {len(all_preds):,} predictions")
        return all_preds, all_probs

    # ── overall metrics ───────────────────────────────────────────────

    def _compute_overall_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_prob: list[float],
    ) -> dict:
        report = classification_report(
            y_true, y_pred,
            target_names=["<=50K", ">50K"],
            output_dict=True,
        )
        cm      = confusion_matrix(y_true, y_pred).tolist()
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_pr  = average_precision_score(y_true, y_prob)

        metrics = {
            "accuracy":          round(float(report["accuracy"]), 4),
            "roc_auc":           round(float(roc_auc), 4),
            "average_precision": round(float(avg_pr), 4),
            "confusion_matrix":  cm,
            "classification_report": {
                k: {mk: round(float(mv), 4) for mk, mv in v.items()}
                if isinstance(v, dict) else round(float(v), 4)
                for k, v in report.items()
            },
        }

        logger.info(f"Overall accuracy : {metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
        logger.info(f"Avg precision    : {metrics['average_precision']:.4f}")
        return metrics

    # ── bias metrics ──────────────────────────────────────────────────

    def _demographic_parity(self, df: pd.DataFrame, attr: str) -> dict:
        """
        Demographic Parity — difference in positive prediction rates
        across groups. Gap must be <= threshold.
        """
        rates   = df.groupby(attr)["prediction"].mean().round(4).to_dict()
        rates   = {k: float(v) for k, v in rates.items()}
        max_gap = round(float(max(rates.values()) - min(rates.values())), 4)
        passed  = bool(is_within_threshold(
            max_gap, CONFIG.thresholds.demographic_parity, "below"
        ))
        return {
            "positive_rates": rates,
            "max_gap":        max_gap,
            "threshold":      CONFIG.thresholds.demographic_parity,
            "passed":         passed,
        }

    def _disparate_impact(self, df: pd.DataFrame, attr: str) -> dict:
        """
        Disparate Impact — ratio of lowest to highest positive rate.
        Ratio must be >= threshold (0.8 = 80% rule).
        """
        rates    = df.groupby(attr)["prediction"].mean().round(4).to_dict()
        rates    = {k: float(v) for k, v in rates.items()}
        max_rate = max(rates.values())
        min_rate = min(rates.values())
        ratio    = round(float(min_rate / max_rate), 4) if max_rate > 0 else 0.0
        passed   = bool(is_within_threshold(
            ratio, CONFIG.thresholds.disparate_impact, "above"
        ))
        return {
            "positive_rates": rates,
            "ratio":          ratio,
            "threshold":      CONFIG.thresholds.disparate_impact,
            "passed":         passed,
        }

    def _equal_opportunity(self, df: pd.DataFrame, attr: str) -> dict:
        """
        Equal Opportunity — difference in True Positive Rates (recall)
        across groups among actual positives. Gap must be <= threshold.
        """
        tpr_by_group = {}
        for group, gdf in df.groupby(attr):
            positives = gdf[gdf[TARGET_COLUMN] == 1]
            tpr = (
                float(positives["prediction"].mean())
                if len(positives) > 0 else 0.0
            )
            tpr_by_group[str(group)] = round(tpr, 4)

        max_gap = round(
            float(max(tpr_by_group.values()) - min(tpr_by_group.values())), 4
        )
        passed = bool(is_within_threshold(
            max_gap, CONFIG.thresholds.equal_opportunity, "below"
        ))
        return {
            "tpr_by_group": tpr_by_group,
            "max_gap":      max_gap,
            "threshold":    CONFIG.thresholds.equal_opportunity,
            "passed":       passed,
        }

    def _equalized_odds(self, df: pd.DataFrame, attr: str) -> dict:
        """
        Equalized Odds — both TPR and FPR gaps must be within threshold
        across groups.
        """
        tpr_by_group, fpr_by_group = {}, {}

        for group, gdf in df.groupby(attr):
            pos = gdf[gdf[TARGET_COLUMN] == 1]
            neg = gdf[gdf[TARGET_COLUMN] == 0]

            tpr = float(pos["prediction"].mean()) if len(pos) > 0 else 0.0
            fpr = float(neg["prediction"].mean()) if len(neg) > 0 else 0.0

            tpr_by_group[str(group)] = round(tpr, 4)
            fpr_by_group[str(group)] = round(fpr, 4)

        tpr_gap = round(float(max(tpr_by_group.values()) - min(tpr_by_group.values())), 4)
        fpr_gap = round(float(max(fpr_by_group.values()) - min(fpr_by_group.values())), 4)
        passed  = bool(
            is_within_threshold(tpr_gap, CONFIG.thresholds.equal_opportunity, "below")
            and
            is_within_threshold(fpr_gap, CONFIG.thresholds.equal_opportunity, "below")
        )
        return {
            "tpr_by_group": tpr_by_group,
            "fpr_by_group": fpr_by_group,
            "tpr_gap":      tpr_gap,
            "fpr_gap":      fpr_gap,
            "threshold":    CONFIG.thresholds.equal_opportunity,
            "passed":       passed,
        }

    def _compute_bias_metrics(self, df: pd.DataFrame) -> tuple[dict, bool, list[str]]:
        all_results      = {}
        all_passed       = True
        flagged_attrs    = []

        for attr in PROTECTED_ATTRIBUTES:
            if attr not in df.columns:
                logger.warning(f"Attribute '{attr}' not in data — skipping")
                continue

            logger.info(f"Computing bias metrics for: {attr}")
            results = {
                "demographic_parity": self._demographic_parity(df, attr),
                "disparate_impact":   self._disparate_impact(df, attr),
                "equal_opportunity":  self._equal_opportunity(df, attr),
                "equalized_odds":     self._equalized_odds(df, attr),
            }

            attr_passed = all(v["passed"] for v in results.values())
            results["attribute_passed"] = attr_passed

            if not attr_passed:
                all_passed = False
                flagged_attrs.append(attr)
                failed = [k for k, v in results.items() if isinstance(v, dict) and not v.get("passed", True)]
                logger.warning(f"Bias detected for '{attr}' — failed checks: {failed}")
            else:
                logger.info(f"All bias checks passed for '{attr}' ✅")

            all_results[attr] = results

        return all_results, all_passed, flagged_attrs

    # ── public entry point ────────────────────────────────────────────

    def run(self) -> EvaluationArtifact:
        logger.info("=" * 50)
        logger.info("Starting Evaluation Component")

        create_directories([str(METRICS_DIR)])

        try:
            # 1 — load data
            if not Path(self.data_path).exists():
                raise FileNotFoundError(
                    f"Processed data not found: {self.data_path}\n"
                    "  → Run DataIngestion stage first"
                )
            df = load_dataframe(self.data_path)
            logger.info(f"Data loaded — shape: {df.shape}")

            # 2 — load model
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}\n"
                    "  → Run PrepareBaseModel stage first"
                )
            model, tokenizer = self._load_model_and_tokenizer()

            # 3 — build text inputs & run inference
            texts              = self._build_texts(df)
            predictions, probs = self._run_inference(model, tokenizer, texts)
            df["prediction"]   = predictions
            df["prob_gt50k"]   = probs

            # 4 — overall metrics
            overall = self._compute_overall_metrics(
                df[TARGET_COLUMN].tolist(), predictions, probs
            )

            # 5 — bias metrics
            bias_results, all_passed, flagged = self._compute_bias_metrics(df)

            # 6 — save results
            save_json(overall,      self.metrics_path)
            save_json(bias_results, self.bias_report_path)

            artifact = EvaluationArtifact(
                metrics_path=self.metrics_path,
                bias_report_path=self.bias_report_path,
                overall_accuracy=overall["accuracy"],
                roc_auc=overall["roc_auc"],
                all_bias_checks_passed=all_passed,
                flagged_attributes=flagged,
            )

            logger.info(f"Evaluation complete: {artifact}")
            return artifact

        except FileNotFoundError as e:
            logger.error(f"Missing file: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error (check torch/CUDA): {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in Evaluation.run(): {e}")
            raise


# ── standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    component = Evaluation()
    artifact  = component.run()

    print("\n── Artifact ────────────────────────────────────────")
    print(f"  Metrics path         : {artifact.metrics_path}")
    print(f"  Bias report path     : {artifact.bias_report_path}")
    print(f"  Overall accuracy     : {artifact.overall_accuracy:.4f}")
    print(f"  ROC-AUC              : {artifact.roc_auc:.4f}")
    print(f"  All bias checks passed: {artifact.all_bias_checks_passed}")
    print(f"  Flagged attributes   : {artifact.flagged_attributes or 'None'}")