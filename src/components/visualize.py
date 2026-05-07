import sys
from pathlib import Path
from typing import Optional, Callable
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from src.config.configuration import load_config
from src.constants import TARGET_COLUMN, PROTECTED_ATTRIBUTES, REPORTS_DIR
from src import get_logger

logger, CONFIG = get_logger(__name__), load_config()
PALETTE = {
    "positive": "#4CAF50", "negative": "#F44336", "neutral": "#5C6BC0",
    "warning": "#FF9800", "pass": "#4CAF50", "fail": "#F44336", "bar": "#5C6BC0",
}
sns.set_theme(style="whitegrid", font_scale=1.1)

def _save(fig: plt.Figure, filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved → {path}")
    return path

def _error_handler(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    return wrapper

def _annotate_bars(ax: plt.Axes, bars: list, values: list, fmt: str = "{:.1%}") -> None:
    """Annotate bars in O(n) time."""
    for bar, val in zip(bars, values):
        if val > 0:
            ax.annotate(fmt.format(val), (bar.get_x() + bar.get_width() / 2, val),
                       ha="center", va="bottom", fontsize=9)


@_error_handler
def plot_class_distribution(df: pd.DataFrame, output_dir: Path = REPORTS_DIR / "charts") -> Path:
    """Class distribution bar + pie chart."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels, colors = ["<=50K", ">50K"], [PALETTE["negative"], PALETTE["positive"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Class Distribution", fontsize=14, fontweight="bold")
    
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Count per class")
    axes[0].set_ylabel("Row count")
    _annotate_bars(axes[0], bars, counts.values, fmt="{:,}")
    
    axes[1].pie(counts.values, labels=[f"{l}\n{c:,}" for l, c in zip(labels, counts.values)],
               autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Class balance")
    
    ratio = counts.max() / counts.min()
    fig.text(0.5, -0.02,
            f"Imbalance ratio: {ratio:.2f}x" + (" — consider class weights" if ratio > 3 else " — acceptable"),
            ha="center", fontsize=10, color=PALETTE["warning"] if ratio > 3 else PALETTE["positive"])
    plt.tight_layout()
    return _save(fig, "class_distribution.png", output_dir)

@_error_handler
def plot_attribute_group_distribution(df: pd.DataFrame, output_dir: Path = REPORTS_DIR / "charts") -> list[Path]:
    """Generate distribution chart for each protected attribute - O(n log n) via sort."""
    saved = []
    for attr in PROTECTED_ATTRIBUTES:
        if attr not in df.columns:
            logger.warning(f"'{attr}' not in DataFrame")
            continue
        counts = df[attr].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, len(counts) * 1.5), 5))
        bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE["bar"], edgecolor="white")
        ax.set_title(f"Group distribution — {attr}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        
        for bar, count in zip(bars, counts.values):
            pct = count / len(df) * 100
            ax.annotate(f"{count:,}\n({pct:.1f}%)", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        saved.append(_save(fig, f"dist_{attr.replace('.', '_')}.png", output_dir))
    return saved

@_error_handler
def plot_rate_by_attribute(df: pd.DataFrame, target_col: str = TARGET_COLUMN, 
                          metric_name: str = "Income Rate", metric_title: str = "",
                          output_dir: Path = REPORTS_DIR / "charts") -> list[Path]:
    """Generic rate-by-attribute plot - O(n log n) via groupby sort."""
    saved = []
    for attr in PROTECTED_ATTRIBUTES:
        if attr not in df.columns:
            continue
        rates = df.groupby(attr)[target_col].mean().sort_values(ascending=False)
        overall = df[target_col].mean()
        max_gap = rates.max() - rates.min()
        threshold = CONFIG.thresholds.demographic_parity
        passed = max_gap <= threshold
        colors = [PALETTE["pass" if abs(r - overall) <= threshold / 2 else "fail"] for r in rates.values]
        
        fig, ax = plt.subplots(figsize=(max(8, len(rates) * 1.5), 5))
        bars = ax.bar(rates.index.astype(str), rates.values, color=colors, edgecolor="white")
        ax.axhline(overall, color="black", linestyle="--", linewidth=1.2, label=f"Overall avg: {overall:.2%}")
        ax.set_title(f"{metric_title or metric_name} by {attr}\nGap: {max_gap:.4f}  |  {'✅ PASS' if passed else '⚠️ FAIL'}",
                    fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, min(1.0, rates.max() + 0.15))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)
        _annotate_bars(ax, bars, rates.values, fmt="{:.1%}")
        plt.tight_layout()
        saved.append(_save(fig, f"{metric_name.lower().replace(' ', '_')}_{attr.replace('.', '_')}.png", output_dir))
    return saved

@_error_handler
def plot_confusion_matrix(y_true: list[int], y_pred: list[int], 
                         output_dir: Path = REPORTS_DIR / "charts") -> Path:
    """Confusion matrix heatmap - O(n) for CM computation."""
    cm = confusion_matrix(y_true, y_pred)
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = np.array([[f"{v}\n({p:.1%})" for v, p in zip(row_v, row_p)]
                       for row_v, row_p in zip(cm, pct)])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax,
               xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], linewidths=0.5)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    return _save(fig, "confusion_matrix.png", output_dir)

@_error_handler
def plot_roc_curve(y_true: list[int], y_prob: list[float], 
                  output_dir: Path = REPORTS_DIR / "charts") -> Path:
    """ROC curve - O(n log n) via sort in roc_curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["neutral"], linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["neutral"])
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    return _save(fig, "roc_curve.png", output_dir)

@_error_handler
def plot_bias_summary(bias_report: dict, output_dir: Path = REPORTS_DIR / "charts") -> Path:
    """Bias metrics heatmap - O(n log n) via dataframe ops."""
    metrics = ["demographic_parity", "disparate_impact", "equal_opportunity", "equalized_odds"]
    attrs = list(bias_report.keys())
    matrix = pd.DataFrame({m: [1 if bias_report[a].get(m, {}).get("passed", False) else 0 for a in attrs]
                          for m in metrics}, index=attrs)
    values = pd.DataFrame({m: [str(bias_report[a].get(m, {}).get("max_gap" if "gap" in str(bias_report[a].get(m, {})) else "ratio", "N/A"))[:6] for a in attrs]
                          for m in metrics}, index=attrs)
    
    cmap = matplotlib.colors.ListedColormap([PALETTE["fail"], PALETTE["pass"]])
    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2.5), max(4, len(attrs) * 1.2)))
    sns.heatmap(matrix.astype(float), annot=values, fmt="", cmap=cmap, vmin=0, vmax=1, ax=ax,
               linewidths=1, linecolor="white", cbar=False,
               xticklabels=[m.replace("_", "\n") for m in metrics], yticklabels=attrs)
    ax.set_title("Bias Audit Summary — PASS / FAIL per metric", fontsize=13, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=PALETTE["pass"], label="PASS"),
                      mpatches.Patch(color=PALETTE["fail"], label="FAIL")],
             loc="upper right", bbox_to_anchor=(1.12, 1))
    plt.tight_layout()
    return _save(fig, "bias_summary.png", output_dir)

@_error_handler
def generate_all_charts(df: pd.DataFrame, y_true: list[int], y_pred: list[int],
                       y_prob: list[float], bias_report: dict,
                       output_dir: Optional[Path] = None) -> dict:
    """Generate all charts efficiently - O(n log n) overall via sorting operations."""
    out = output_dir or (REPORTS_DIR / "charts")
    logger.info(f"Generating all charts → {out}")
    
    charts = {
        "class_distribution": plot_class_distribution(df, out),
        "attribute_dists": plot_attribute_group_distribution(df, out),
        "income_rates": plot_rate_by_attribute(df, TARGET_COLUMN, "Income Rate", "Income rate (>50K)", out),
        "confusion_matrix": plot_confusion_matrix(y_true, y_pred, out),
        "roc_curve": plot_roc_curve(y_true, y_prob, out),
        "bias_summary": plot_bias_summary(bias_report, out),
        "demographic_parity": plot_rate_by_attribute(df, "prediction", "Positive Prediction Rate", "Demographic Parity", out) if "prediction" in df.columns else [],
        "equal_opportunity": plot_equal_opportunity(df, out) if "prediction" in df.columns else [],
    }
    logger.info(f"All charts generated ✅ — {len(charts)} chart groups")
    return charts

def plot_equal_opportunity(df: pd.DataFrame, output_dir: Path = REPORTS_DIR / "charts") -> list[Path]:
    """Equal opportunity (TPR) per group - O(n log n) via groupby."""
    saved = []
    for attr in PROTECTED_ATTRIBUTES:
        if attr not in df.columns:
            continue
        tpr_dict = {str(g): df[df[attr] == g][df[df[attr] == g][TARGET_COLUMN] == 1]["prediction"].mean() or 0
                   for g in df[attr].unique()}
        tpr_s = pd.Series(tpr_dict).sort_values(ascending=False)
        max_gap = tpr_s.max() - tpr_s.min()
        passed = max_gap <= CONFIG.thresholds.equal_opportunity
        colors = [PALETTE["pass" if passed else "fail"]] * len(tpr_s)
        
        fig, ax = plt.subplots(figsize=(max(8, len(tpr_s) * 1.5), 5))
        bars = ax.bar(tpr_s.index, tpr_s.values, color=colors, edgecolor="white")
        ax.set_title(f"Equal Opportunity (TPR) — {attr}\nGap: {max_gap:.4f}  {'✅ PASS' if passed else '⚠️ FAIL'}",
                    fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate")
        ax.set_ylim(0, min(1.0, tpr_s.max() + 0.15))
        ax.tick_params(axis="x", rotation=30)
        _annotate_bars(ax, bars, tpr_s.values, fmt="{:.1%}")
        plt.tight_layout()
        saved.append(_save(fig, f"eo_{attr.replace('.', '_')}.png", output_dir))
    return saved