from pathlib import Path

# ── file paths ───────────────────────────────────────────────────────
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# ── directory paths ──────────────────────────────────────────────────
DATA_DIR          = Path("data")
RAW_DATA_DIR      = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUTS_DIR       = Path("outputs")
LOGS_DIR          = Path("logs")
MODEL_DIR         = Path("outputs/model")
TOKENIZER_DIR     = Path("outputs/tokenizer")
METRICS_DIR       = Path("outputs/metrics")
REPORTS_DIR       = Path("outputs/reports")

# ── file names ───────────────────────────────────────────────────────
RAW_DATA_FILE       = RAW_DATA_DIR      / "adult_income.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "clean.csv"
METRICS_FILE        = METRICS_DIR       / "metrics.json"
REPORT_FILE         = REPORTS_DIR       / "bias_report.pdf"

# ── model constants ──────────────────────────────────────────────────
TARGET_COLUMN       = "income"
TARGET_MAPPING      = {"<=50K": 0, ">50K": 1}
MISSING_VALUE_TOKEN = "?"

# ── protected attributes ─────────────────────────────────────────────
PROTECTED_ATTRIBUTES = ["sex", "race", "marital.status"]

# ── pipeline stage names ─────────────────────────────────────────────
STAGE_01 = "Stage 01 — Data Ingestion"
STAGE_02 = "Stage 02 — Prepare Base Model"
STAGE_03 = "Stage 03 — Bias Metrics"