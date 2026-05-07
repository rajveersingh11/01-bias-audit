# Bias Audit: Fairness Evaluation Pipeline

A machine learning pipeline that audits algorithmic bias in income prediction models. This project evaluates model fairness across demographic attributes and generates comprehensive reports on disparate impact, demographic parity, and equal opportunity violations.

## Overview

This project implements a comprehensive bias audit system for machine learning models, focusing on the income prediction task. It:

- **Ingests and cleans** datasets with missing value handling
- **Prepares and loads** a pre-trained DistilBERT model for inference
- **Evaluates** model performance and computes bias metrics across protected attributes (sex, race, marital status)
- **Generates** detailed reports with visualizations and PDF summaries

### Bias Metrics

The pipeline computes the following fairness metrics per demographic attribute:

- **Demographic Parity (DP)**: Are positive predictions equally likely across groups?
- **Disparate Impact (DI)**: What's the ratio of favorable outcomes between protected and advantaged groups?
- **Equal Opportunity (EO)**: Are true positive rates equal across groups?
- **Equalized Odds (EqO)**: Are both TPR and FPR equal across groups?

## Project Structure

```
01-bias-audit/
├── main.py                       # Orchestrates all pipeline stages
├── dvc.yaml                      # DVC pipeline configuration
├── params.yaml                   # Model and evaluation hyperparameters
├── requirements.txt              # Python dependencies
├── config/
│   ├── config.yaml              # Configuration schema
│   └── configuration.py          # Config loader
├── data/
│   ├── raw/                     # Raw input datasets
│   └── processed/               # Cleaned and processed data
├── outputs/
│   ├── model/                   # Loaded pre-trained model
│   ├── tokenizer/               # Model tokenizer
│   ├── metrics/                 # JSON metric reports
│   ├── cache/                   # Cached predictions
│   └── reports/
│       ├── charts/              # PNG visualizations
│       └── report.pdf           # Final PDF report
├── src/
│   ├── components/              # Core processing logic
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── evaluation.py
│   │   ├── report.py
│   │   └── visualize.py
│   ├── pipeline/                # Orchestration layer
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_evaluation.py
│   │   └── stage_04_report.py
│   ├── config/
│   │   ├── config.yaml
│   │   └── configuration.py
│   ├── constants/
│   │   └── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── common.py
│   └── __init__.py
└── tests/                       # Unit tests
```

## Pipeline Stages

### Stage 1: Data Ingestion
- Downloads and caches the Adult Income dataset
- Handles missing values (drops rows with NaN)
- Encodes target variable (income level)
- Validates class distribution and group sizes
- Outputs: Cleaned CSV with preprocessed features

### Stage 2: Prepare Base Model
- Loads pre-trained DistilBERT model from Hugging Face
- Caches model and tokenizer locally
- Configures for binary classification (income prediction)
- Outputs: Model and tokenizer directories

### Stage 3: Evaluation
- Runs inference on all samples
- Computes accuracy, ROC-AUC, average precision
- Calculates bias metrics for each demographic attribute
- Flags attributes that violate fairness thresholds
- Outputs: `metrics.json` (performance) and `bias_report.json` (fairness)

### Stage 4: Report Generation
- Generates comprehensive visualizations (distributions, confusion matrix, ROC curve)
- Creates bias summary charts per attribute
- Builds PDF report with executive summary and detailed findings
- Outputs: PDF report and chart directory

## Getting Started

### Prerequisites

- Python 3.12+
- pip or conda
- ~5GB disk space (for model and data)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rajveersingh11/01-bias-audit.git
   cd 01-bias-audit
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

#### Option 1: Run all stages
```bash
python main.py
```

#### Option 2: Use DVC for reproducibility
```bash
dvc repro
```

#### Option 3: Run individual stages
```bash
python src/pipeline/stage_01_data_ingestion.py
python src/pipeline/stage_02_prepare_base_model.py
python src/pipeline/stage_03_evaluation.py
python src/pipeline/stage_04_report.py
```

### Configuration

Edit `params.yaml` to customize:

- **Model parameters**: epochs, learning rate, warmup steps
- **Data parameters**: train/test split, random seed
- **Bias thresholds**: minimum group size, confidence level
- **Evaluation parameters**: batch size, device (cpu/cuda)

Example:
```yaml
model:
  epochs: 3
  learning_rate: 2e-5

bias:
  min_group_size: 100      # Skip small demographic groups
```

## Outputs

### Metrics (`outputs/metrics/metrics.json`)
```json
{
  "overall_accuracy": 0.7011,
  "roc_auc": 0.5531,
  "avg_precision": 0.2794
}
```

### Bias Report (`outputs/metrics/bias_report.json`)
```json
{
  "sex": {
    "demographic_parity": 0.0077,      // PASS (< 0.1)
    "disparate_impact": 0.9403,        // PASS (> 0.8)
    "equal_opportunity": 0.0549,       // PASS (< 0.1)
    "equalized_odds": 0.0549           // PASS (< 0.1)
  },
  "race": {
    "demographic_parity": 0.5329,      // FAIL
    "disparate_impact": 0.1135,        // FAIL
    ...
  }
}
```

### Report (`outputs/reports/report.pdf`)
- Executive summary with bias verdict
- Performance metrics
- Distribution charts per demographic
- Income rate analysis
- Confusion matrix and ROC curve
- Bias severity heatmaps

### Charts (`outputs/reports/charts/`)
- `class_distribution.png` - Target variable balance
- `dist_*.png` - Feature distributions by demographic
- `income_rate_*.png` - Positive outcome rates
- `confusion_matrix.png` - Model prediction accuracy
- `roc_curve.png` - ROC curve
- `bias_summary.png` - Overall fairness status
- `eo_*.png`, `positive_prediction_rate_*.png` - Fairness metric visualizations

## Docker Usage

### Build the image
```bash
docker build -t bias-audit:latest .
```

### Run the pipeline in a container
```bash
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  bias-audit:latest
```

### Interactive shell
```bash
docker run --rm -it \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  bias-audit:latest bash
```

## Key Features

- **Reproducible Pipeline**: DVC and seeds ensure consistent results
- **Comprehensive Bias Metrics**: Multiple fairness definitions per protected attribute
- **Automated Report Generation**: PDF reports with visualizations
- **Modular Architecture**: Easy to extend with new fairness metrics
- **Model Optimization**: int8 quantization and torch.compile for faster inference
- **Class Imbalance Handling**: Detects and warns about imbalanced datasets
- **Small Group Detection**: Flags demographic groups below minimum size threshold

## Bias Detection Example

When bias is detected, the pipeline logs warnings:

```
WARNING: Bias detected for 'race' — failed checks: 
  ['demographic_parity', 'disparate_impact', 'equal_opportunity', 'equalized_odds']
```

The PDF report highlights these findings with severity indicators and recommendations.

## Performance Considerations

- **GPU Acceleration**: Set `evaluation.device: cuda` in `params.yaml` for faster inference
- **Model Caching**: First run downloads model (~500MB); subsequent runs use cache
- **Batch Processing**: Adjust `evaluation.batch_size` based on available memory
- **Data Sampling**: Set `data.max_samples` to test on subset before full run

## Dependencies

Key packages:
- `torch` & `transformers` - Deep learning and pre-trained models
- `pandas` & `scikit-learn` - Data processing and metrics
- `matplotlib` & `seaborn` - Visualizations
- `fpdf2` - PDF report generation
- `dvc` - Pipeline versioning and reproducibility
- `pyyaml` - Configuration management

See `pyproject.toml` for complete dependency list.

## Troubleshooting

### "No module named 'src'" error
Ensure you're running from the project root:
```bash
cd 01-bias-audit
python main.py
```

### Out of memory during inference
Reduce batch size in `params.yaml`:
```yaml
evaluation:
  batch_size: 32  # Reduce from default
```

### Missing pre-trained model
First run will download the model. Ensure internet connection and ~1GB free space.

### Font warnings in charts
Unicode glyph warnings are non-fatal and don't affect output.

## References

- [Fairness definitions and their politics](https://arxiv.org/abs/1809.04578)
- [Disparate Impact](https://en.wikipedia.org/wiki/Disparate_impact)
- [Equal Opportunity in ML](https://arxiv.org/abs/1610.02413)
- [DVC Documentation](https://dvc.org/doc)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## Author

Created as a portfolio project for demonstrating ML fairness and bias audit practices.

## License

MIT License - See LICENSE file for details.
