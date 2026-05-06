from dataclasses import dataclass
from pathlib import Path
from urllib import request as url_request

import pandas as pd
from datasets import load_dataset

from config.configuration import load_config
from src.constants import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    MISSING_VALUE_TOKEN,
    PROTECTED_ATTRIBUTES,
)
from src.utils import (
    create_directories,
    save_dataframe,
    load_dataframe,
    get_dataframe_info,
    validate_columns,
    encode_target,
    get_size,
)
from src import get_logger

logger = get_logger(__name__)
CONFIG = load_config()

REQUIRED_COLUMNS = [
    "age",
    "education",
    "occupation",
    "sex",
    "race",
    "income",
    "hours.per.week",
    "marital.status",
    "workclass",
    "relationship",
    "capital.gain",
    "capital.loss",
    "native.country",
    "fnlwgt",
    "education.num",
]


@dataclass
class DataIngestionArtifact:
    raw_data_path: str
    processed_data_path: str
    row_count: int
    column_count: int
    dropped_rows: int
    class_distribution: dict


class DataIngestion:
    def __init__(self):
        self.source_url     = getattr(CONFIG.data, "source_url", None)  # optional URL
        self.source_hf      = CONFIG.data.source                        # HuggingFace dataset id
        self.split          = CONFIG.data.split
        self.raw_path       = Path(RAW_DATA_FILE)
        self.processed_path = Path(PROCESSED_DATA_FILE)

    # ── private ───────────────────────────────────────────────────────

    def _download_from_url(self) -> None:
        """Download raw CSV directly from a URL using urllib."""
        logger.info(f"Downloading from URL: {self.source_url}")

        filename, headers = url_request.urlretrieve(
            url=self.source_url,
            filename=str(self.raw_path),
        )
        logger.info(f"{filename} downloaded with following info:\n{headers}")

    def _download_from_huggingface(self) -> None:
        """Download from HuggingFace datasets and save as CSV."""
        logger.info(f"Downloading from HuggingFace: {self.source_hf}")

        try:
            ds = load_dataset(self.source_hf, split=self.split)
            df = ds.to_pandas()
            save_dataframe(df, str(self.raw_path))
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            raise

    def _download(self) -> pd.DataFrame:
        """
        Download raw data — skips if already cached.
        Prefers URL download if source_url is set in config,
        falls back to HuggingFace otherwise.
        """
        if not self.raw_path.exists():
            create_directories([str(self.raw_path.parent)])

            if self.source_url:
                self._download_from_url()
            else:
                self._download_from_huggingface()
        else:
            logger.info(
                f"File already exists of size: {get_size(self.raw_path)}"
            )

        return load_dataframe(str(self.raw_path))

    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all string columns."""
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
        logger.debug(f"Whitespace stripped from {len(str_cols)} string columns")
        return df

    def _drop_missing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Replace '?' tokens with NaN and drop affected rows."""
        before = len(df)
        df = df.replace(MISSING_VALUE_TOKEN, pd.NA).dropna()
        dropped = before - len(df)
        pct     = dropped / before * 100
        logger.info(f"Dropped {dropped:,} rows with missing values ({pct:.2f}%)")

        if pct > 10.0:
            logger.warning(
                f"{pct:.2f}% of rows dropped — higher than expected. "
                "Review MISSING_VALUE_TOKEN or upstream data quality."
            )
        return df.reset_index(drop=True), dropped

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map income strings → 0 / 1 integers."""
        df = encode_target(df, TARGET_COLUMN, TARGET_MAPPING)
        logger.info(
            f"Target '{TARGET_COLUMN}' encoded: "
            f"{df[TARGET_COLUMN].value_counts().to_dict()}"
        )
        return df

    def _check_class_balance(self, df: pd.DataFrame) -> dict:
        """Log a warning if class imbalance ratio exceeds 3x."""
        counts = df[TARGET_COLUMN].value_counts().to_dict()
        ratio  = max(counts.values()) / min(counts.values())

        if ratio > 3.0:
            logger.warning(
                f"High class imbalance — ratio {ratio:.2f}x. "
                "Consider class weights or oversampling in model training."
            )
        else:
            logger.info(f"Class balance ratio: {ratio:.2f}x  {counts}")

        return counts

    def _check_group_sizes(self, df: pd.DataFrame) -> None:
        """Warn if any protected attribute group is too small for reliable metrics."""
        min_size = CONFIG.params.min_group_size
        for attr in PROTECTED_ATTRIBUTES:
            if attr not in df.columns:
                logger.warning(f"Protected attribute '{attr}' not found in data")
                continue
            counts = df[attr].value_counts()
            small  = counts[counts < min_size]
            if not small.empty:
                logger.warning(
                    f"Attribute '{attr}' has groups below min_group_size={min_size}: "
                    f"{small.to_dict()} — bias metrics may be unreliable for these groups."
                )

    # ── public ───────────────────────────────────────────────────────

    def download(self) -> pd.DataFrame:
        """Public entry point for download step only (useful in notebooks)."""
        create_directories([str(self.raw_path.parent)])
        return self._download()

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Public entry point for cleaning step only.
        Returns (cleaned_df, dropped_row_count).
        """
        logger.info(f"Cleaning started — shape before: {df.shape}")
        validate_columns(df, REQUIRED_COLUMNS)

        df           = self._strip_whitespace(df)
        df, dropped  = self._drop_missing(df)
        df           = self._encode_target(df)

        info = get_dataframe_info(df)
        logger.info(f"Cleaning complete — shape after: {info['shape']}")
        logger.debug(f"Null counts after clean: {info['null_counts']}")
        return df, dropped

    def run(self) -> DataIngestionArtifact:
        logger.info("=" * 50)
        logger.info("Starting Data Ingestion Component")

        # guarantee directories exist
        create_directories([
            str(self.raw_path.parent),
            str(self.processed_path.parent),
        ])

        try:
            # step 1 — download / load from cache
            df_raw = self._download()

            # step 2 — clean
            df_clean, dropped = self.clean(df_raw)

            # step 3 — quality checks
            class_dist = self._check_class_balance(df_clean)
            self._check_group_sizes(df_clean)

            # step 4 — save
            save_dataframe(df_clean, str(self.processed_path))

            artifact = DataIngestionArtifact(
                raw_data_path=str(self.raw_path),
                processed_data_path=str(self.processed_path),
                row_count=df_clean.shape[0],
                column_count=df_clean.shape[1],
                dropped_rows=dropped,
                class_distribution=class_dist,
            )

            logger.info(f"Data Ingestion complete: {artifact}")
            return artifact

        except FileNotFoundError as e:
            logger.error(f"File not found during ingestion: {e}")
            raise
        except ValueError as e:
            logger.error(f"Data validation failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in DataIngestion.run(): {e}")
            raise


# ── standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    component = DataIngestion()
    artifact  = component.run()

    print("\n── Artifact ────────────────────────────────────")
    print(f"  Raw path          : {artifact.raw_data_path}")
    print(f"  Processed path    : {artifact.processed_data_path}")
    print(f"  Rows              : {artifact.row_count:,}")
    print(f"  Columns           : {artifact.column_count}")
    print(f"  Dropped rows      : {artifact.dropped_rows:,}")
    print(f"  Class distribution: {artifact.class_distribution}")