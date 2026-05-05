import yaml
from pathlib import Path
from dataclasses import dataclass
from src.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    MODEL_DIR,
    TOKENIZER_DIR,
    METRICS_FILE,
    REPORT_FILE,
)


@dataclass
class DataConfig:
    raw_path: Path
    processed_path: Path
    source: str
    split: str


@dataclass
class ModelConfig:
    name: str
    batch_size: int
    max_length: int


@dataclass
class ReportConfig:
    output_path: Path
    title: str
    author: str


@dataclass
class ThresholdConfig:
    demographic_parity: float
    disparate_impact: float
    equal_opportunity: float


@dataclass
class ParamsConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    test_size: float
    val_size: float
    random_seed: int
    max_samples: int | None
    min_group_size: int
    confidence_level: float
    batch_size: int
    device: str


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    report: ReportConfig
    thresholds: ThresholdConfig
    params: ParamsConfig
    protected_attributes: list[str]


def load_config(
    config_path: Path = CONFIG_FILE_PATH,
    params_path: Path = PARAMS_FILE_PATH,
) -> AppConfig:

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    return AppConfig(
        data=DataConfig(
            raw_path=RAW_DATA_FILE,
            processed_path=PROCESSED_DATA_FILE,
            source=raw["data"]["source"],
            split=raw["data"]["split"],
        ),
        model=ModelConfig(
            name=raw["model"]["name"],
            batch_size=raw["model"]["batch_size"],
            max_length=raw["model"]["max_length"],
        ),
        report=ReportConfig(
            output_path=REPORT_FILE,
            title=raw["report"]["title"],
            author=raw["report"]["author"],
        ),
        thresholds=ThresholdConfig(**raw["thresholds"]),
        params=ParamsConfig(
            epochs=params["model"]["epochs"],
            learning_rate=params["model"]["learning_rate"],
            weight_decay=params["model"]["weight_decay"],
            warmup_steps=params["model"]["warmup_steps"],
            test_size=params["data"]["test_size"],
            val_size=params["data"]["val_size"],
            random_seed=params["data"]["random_seed"],
            max_samples=params["data"]["max_samples"],
            min_group_size=params["bias"]["min_group_size"],
            confidence_level=params["bias"]["confidence_level"],
            batch_size=params["evaluation"]["batch_size"],
            device=params["evaluation"]["device"],
        ),
        protected_attributes=raw["protected_attributes"],
    )


CONFIG = load_config()