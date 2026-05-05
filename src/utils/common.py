# src/utils/common.py

import os
import yaml
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any
from src import get_logger
from src.constants import (
    LOGS_DIR,
    OUTPUTS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
)

logger = get_logger(__name__)


# ── directory helpers ────────────────────────────────────────────────

def create_directories(paths: list[str]) -> None:
    """Create multiple directories at once, skip if already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created: {path}")


def get_project_root() -> Path:
    """Return absolute path to project root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def get_all_output_dirs() -> list[Path]:
    """Return all output directories — useful for initialising project."""
    return [LOGS_DIR, OUTPUTS_DIR, METRICS_DIR, REPORTS_DIR]


def init_project_dirs() -> None:
    """
    Call once at startup in main.py to guarantee
    all output folders exist before pipeline runs.
    """
    create_directories([str(p) for p in get_all_output_dirs()])
    logger.info("All output directories initialised")


# ── file I/O ─────────────────────────────────────────────────────────

def save_yaml(data: dict, path: str) -> None:
    create_directories([str(Path(path).parent)])
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"YAML saved: {path}")


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    logger.debug(f"YAML loaded: {path}")
    return data


def save_json(data: dict, path: str) -> None:
    create_directories([str(Path(path).parent)])
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved: {path}")


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    logger.debug(f"JSON loaded: {path}")
    return data


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    create_directories([str(Path(path).parent)])
    df.to_csv(path, index=False)
    logger.info(f"DataFrame saved: {path} — shape: {df.shape}")


def load_dataframe(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"DataFrame loaded: {path} — shape: {df.shape}")
    return df


def save_model(obj: Any, path: str) -> None:
    create_directories([str(Path(path).parent)])
    joblib.dump(obj, path)
    logger.info(f"Model saved: {path}")


def load_model_artifact(path: str) -> Any:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    obj = joblib.load(path)
    logger.info(f"Model artifact loaded: {path}")
    return obj


# ── data helpers ─────────────────────────────────────────────────────

def get_dataframe_info(df: pd.DataFrame) -> dict:
    return {
        "shape":          df.shape,
        "columns":        df.columns.tolist(),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "null_counts":    df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def encode_target(df: pd.DataFrame, column: str,
                  mapping: dict) -> pd.DataFrame:
    df[column] = df[column].map(mapping)
    if df[column].isnull().any():
        raise ValueError(
            f"Unmapped values in '{column}'. Check mapping: {mapping}"
        )
    return df


# ── metric helpers ───────────────────────────────────────────────────

def is_within_threshold(value: float, threshold: float,
                        direction: str = "below") -> bool:
    result = value <= threshold if direction == "below" else value >= threshold
    status = "PASS" if result else "FAIL"
    logger.info(f"Threshold check [{status}] value={value:.4f} "
                f"threshold={threshold} direction={direction}")
    return result