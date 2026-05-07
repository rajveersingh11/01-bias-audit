from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json

from fpdf import FPDF, XPos, YPos

from config.configuration import load_config
from src.constants import (
    REPORTS_DIR,
    METRICS_DIR,
    PROTECTED_ATTRIBUTES,
    TARGET_COLUMN,
)
from src.utils import load_json, create_directories
from src import get_logger
from src.components.visualize import generate_all_charts

logger = get_logger(__name__)
CONFIG = load_config()


# ── artifact ──────────────────────────────────────────────────────────

@dataclass
class ReportArtifact:
    report_path: str
    charts_dir: str
    all_bias_checks_passed: bool
    generated_at: str


# ── PDF class ─────────────────────────────────────────────────────────

class BiasAuditPDF(FPDF):
    """Custom FPDF subclass with header/footer and reusable layout helpers."""

    def __init__(self, title: str, author: str):
        super().__init__()
        self.report_title  = title
        self.report_author = author
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=15)

    # ── header / footer ───────────────────────────────────────────────

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, self.report_title, align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 200, 200)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5,
                  f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d')}",
                  align="C")

    # ── layout helpers ────────────────────────────────────────────────

    def section_title(self, text: str) -> None:
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(240, 240, 245)
        self.cell(0, 9, f"  {text}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def subsection_title(self, text: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def metric_row(self, label: str, value: str,
                   passed: bool | None = None) -> None:
        """Single key-value row with optional PASS/FAIL badge."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.cell(80, 7, label)
        self.set_font("Helvetica", "B", 10)
        self.cell(60, 7, value)

        if passed is not None:
            if passed:
                self.set_fill_color(76, 175, 80)
            else:
                self.set_fill_color(244, 67, 54)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 8)
            badge = "  PASS  " if passed else "  FAIL  "
            self.cell(20, 7, badge, fill=True, align="C")
            self.set_text_color(50, 50, 50)

        self.ln(7)

    def add_chart(self, image_path: str | Path,
                  caption: str = "", width: int = 170) -> None:
        """Embed a chart PNG with optional caption."""
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Chart not found - skipping: {path}")
            return

        if self.get_y() > 220:
            self.add_page()

        x = (210 - width) / 2
        self.image(str(path), x=x, w=width)

        if caption:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, caption, align="C",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def divider(self) -> None:
        self.set_draw_color(220, 220, 220)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)


# ── report builder ────────────────────────────────────────────────────

class ReportGenerator:
    def __init__(
        self,
        metrics_path: str  = str(METRICS_DIR / "metrics.json"),
        bias_path: str     = str(METRICS_DIR / "bias_report.json"),
        output_path: str   = str(REPORTS_DIR / "bias_audit_report.pdf"),
        charts_dir: str    = str(REPORTS_DIR / "charts"),
    ):
        self.metrics_path = metrics_path
        self.bias_path    = bias_path
        self.output_path  = output_path
        self.charts_dir   = Path(charts_dir)

    # ── pages ─────────────────────────────────────────────────────────

    def _cover_page(self, pdf: BiasAuditPDF) -> None:
        pdf.add_page()
        pdf.ln(30)

        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 12, CONFIG.report.title, align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(4)
        pdf.set_font("Helvetica", "", 13)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(0, 8, "Automated Fairness & Bias Audit", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(20)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 7, f"Author   : {CONFIG.report.author}", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 7,
                 f"Date     : {datetime.now().strftime('%B %d, %Y')}",
                 align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 7,
                 f"Model    : {CONFIG.model.name}",
                 align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 7,
                 f"Dataset  : {CONFIG.data.source}",
                 align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(25)
        pdf.divider()
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(
            0, 6,
            "This report was automatically generated by the Bias Audit Pipeline. "
            "It evaluates model fairness across protected attributes including "
            f"{', '.join(PROTECTED_ATTRIBUTES)} using industry-standard metrics: "
            "Demographic Parity, Disparate Impact, Equal Opportunity, and Equalized Odds.",
            align="C",
        )

    def _executive_summary(
        self,
        pdf: BiasAuditPDF,
        metrics: dict,
        bias_report: dict,
        all_passed: bool,
    ) -> None:
        pdf.add_page()
        pdf.section_title("1.  Executive Summary")

        # overall verdict banner
        if all_passed:
            pdf.set_fill_color(76, 175, 80)
            verdict = "ALL BIAS CHECKS PASSED"
        else:
            pdf.set_fill_color(244, 67, 54)
            verdict = "BIAS DETECTED - REVIEW REQUIRED"

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, f"  {verdict}", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(50, 50, 50)
        pdf.ln(4)

        pdf.subsection_title("Model Performance")
        pdf.metric_row("Overall Accuracy",
                       f"{metrics.get('accuracy', 0):.4f}")
        pdf.metric_row("ROC-AUC Score",
                       f"{metrics.get('roc_auc', 0):.4f}")
        pdf.metric_row("Average Precision",
                       f"{metrics.get('average_precision', 0):.4f}")

        pdf.ln(3)
        pdf.subsection_title("Bias Check Results")

        for attr in PROTECTED_ATTRIBUTES:
            if attr not in bias_report:
                continue
            attr_results = bias_report[attr]
            attr_passed  = all(
                v.get("passed", False)
                for v in attr_results.values()
                if isinstance(v, dict) and "passed" in v
            )
            pdf.metric_row(f"  {attr}", "", passed=attr_passed)

    def _model_performance_page(
        self,
        pdf: BiasAuditPDF,
        metrics: dict,
        charts: dict,
    ) -> None:
        pdf.add_page()
        pdf.section_title("2.  Model Performance")

        pdf.subsection_title("Classification Report")
        cr = metrics.get("classification_report", {})
        for label in ["<=50K", ">50K", "macro avg", "weighted avg"]:
            if label in cr and isinstance(cr[label], dict):
                r = cr[label]
                pdf.body_text(
                    f"{label:<20}  "
                    f"precision={r.get('precision', 0):.4f}  "
                    f"recall={r.get('recall', 0):.4f}  "
                    f"f1={r.get('f1-score', 0):.4f}"
                )

        pdf.add_chart(
            self.charts_dir / "confusion_matrix.png",
            "Figure 1 - Confusion Matrix",
        )
        pdf.add_chart(
            self.charts_dir / "roc_curve.png",
            "Figure 2 - ROC Curve",
        )

    def _data_overview_page(
        self,
        pdf: BiasAuditPDF,
        charts: dict,
    ) -> None:
        pdf.add_page()
        pdf.section_title("3.  Data Overview")

        pdf.body_text(
            f"Dataset source  : {CONFIG.data.source}\n"
            f"Protected attributes audited: {', '.join(PROTECTED_ATTRIBUTES)}"
        )

        pdf.add_chart(
            self.charts_dir / "class_distribution.png",
            "Figure 3 - Target Class Distribution",
        )

        for attr in PROTECTED_ATTRIBUTES:
            attr_key = attr.replace(".", "_")
            pdf.add_chart(
                self.charts_dir / f"dist_{attr_key}.png",
                f"Figure - Group distribution: {attr}",
                width=150,
            )
            pdf.add_chart(
                self.charts_dir / f"income_rate_{attr_key}.png",
                f"Figure - Income rate by {attr}",
                width=150,
            )

    def _bias_metrics_page(
        self,
        pdf: BiasAuditPDF,
        bias_report: dict,
        charts: dict,
    ) -> None:
        pdf.add_page()
        pdf.section_title("4.  Bias Metrics Detail")

        pdf.body_text(
            "The following metrics evaluate whether the model treats different "
            "demographic groups equitably. Each check is evaluated against the "
            "configured threshold from params.yaml."
        )

        pdf.add_chart(
            self.charts_dir / "bias_summary.png",
            "Figure - Bias Audit Summary Heatmap",
        )

        for attr in PROTECTED_ATTRIBUTES:
            if attr not in bias_report:
                continue

            pdf.subsection_title(f"Attribute: {attr}")
            attr_data = bias_report[attr]

            for metric_name, metric_data in attr_data.items():
                if not isinstance(metric_data, dict):
                    continue

                passed = metric_data.get("passed", False)

                if metric_name == "demographic_parity":
                    val = metric_data.get("max_gap", 0)
                    pdf.metric_row(
                        "  Demographic Parity gap",
                        f"{val:.4f}  (threshold: {metric_data.get('threshold', 0)})",
                        passed=passed,
                    )
                elif metric_name == "disparate_impact":
                    val = metric_data.get("ratio", 0)
                    pdf.metric_row(
                        "  Disparate Impact ratio",
                        f"{val:.4f}  (threshold: {metric_data.get('threshold', 0)})",
                        passed=passed,
                    )
                elif metric_name == "equal_opportunity":
                    val = metric_data.get("max_gap", 0)
                    pdf.metric_row(
                        "  Equal Opportunity gap",
                        f"{val:.4f}  (threshold: {metric_data.get('threshold', 0)})",
                        passed=passed,
                    )
                elif metric_name == "equalized_odds":
                    tpr_gap = metric_data.get("tpr_gap", 0)
                    fpr_gap = metric_data.get("fpr_gap", 0)
                    pdf.metric_row(
                        "  Equalized Odds (TPR gap)",
                        f"{tpr_gap:.4f}",
                        passed=passed,
                    )
                    pdf.metric_row(
                        "  Equalized Odds (FPR gap)",
                        f"{fpr_gap:.4f}",
                    )

            attr_key = attr.replace(".", "_")
            pdf.add_chart(
                self.charts_dir / f"dp_{attr_key}.png",
                f"Demographic Parity - {attr}",
                width=150,
            )
            pdf.add_chart(
                self.charts_dir / f"eo_{attr_key}.png",
                f"Equal Opportunity - {attr}",
                width=150,
            )
            pdf.divider()

    def _config_page(self, pdf: BiasAuditPDF) -> None:
        pdf.add_page()
        pdf.section_title("5.  Audit Configuration")

        pdf.subsection_title("Bias Thresholds")
        pdf.metric_row("Demographic Parity max gap",
                       str(CONFIG.thresholds.demographic_parity))
        pdf.metric_row("Disparate Impact min ratio",
                       str(CONFIG.thresholds.disparate_impact))
        pdf.metric_row("Equal Opportunity max gap",
                       str(CONFIG.thresholds.equal_opportunity))

        pdf.ln(3)
        pdf.subsection_title("Model Configuration")
        pdf.metric_row("Model name",  CONFIG.model.name)
        pdf.metric_row("Batch size",  str(CONFIG.model.batch_size))
        pdf.metric_row("Max length",  str(CONFIG.model.max_length))
        pdf.metric_row("Device",      CONFIG.params.device)

        pdf.ln(3)
        pdf.subsection_title("Training Parameters")
        pdf.metric_row("Epochs",         str(CONFIG.params.epochs))
        pdf.metric_row("Learning rate",  str(CONFIG.params.learning_rate))
        pdf.metric_row("Random seed",    str(CONFIG.params.random_seed))

    # ── main entry point ──────────────────────────────────────────────

    def run(
        self,
        df,
        y_true: list[int],
        y_pred: list[int],
        y_prob: list[float],
    ) -> ReportArtifact:
        logger.info("=" * 50)
        logger.info("Starting Report Generation Component")

        create_directories([
            str(REPORTS_DIR),
            str(self.charts_dir),
        ])

        try:
            # 1 - load metrics
            if not Path(self.metrics_path).exists():
                raise FileNotFoundError(
                    f"Metrics not found: {self.metrics_path}\n"
                    "  → Run Evaluation stage first"
                )
            if not Path(self.bias_path).exists():
                raise FileNotFoundError(
                    f"Bias report not found: {self.bias_path}\n"
                    "  → Run Evaluation stage first"
                )

            metrics     = load_json(self.metrics_path)
            bias_report = load_json(self.bias_path)

            all_passed = all(
                bias_report[attr].get("attribute_passed", False)
                for attr in bias_report
            )

            # 2 - generate charts
            logger.info("Generating charts...")
            charts = generate_all_charts(
                df, y_true, y_pred, y_prob,
                bias_report,
                output_dir=self.charts_dir,
            )

            # 3 - build PDF
            logger.info("Building PDF report...")
            pdf = BiasAuditPDF(
                title=CONFIG.report.title,
                author=CONFIG.report.author,
            )
            pdf.set_title(CONFIG.report.title)
            pdf.set_author(CONFIG.report.author)

            self._cover_page(pdf, )
            self._executive_summary(pdf, metrics, bias_report, all_passed)
            self._model_performance_page(pdf, metrics, charts)
            self._data_overview_page(pdf, charts)
            self._bias_metrics_page(pdf, bias_report, charts)
            self._config_page(pdf)

            # 4 - save PDF
            output_path = Path(self.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pdf.output(str(output_path))

            size_kb = output_path.stat().st_size / 1024
            logger.info(f"PDF report saved → {output_path}  ({size_kb:.1f} KB)")

            artifact = ReportArtifact(
                report_path=str(output_path),
                charts_dir=str(self.charts_dir),
                all_bias_checks_passed=all_passed,
                generated_at=datetime.now().isoformat(),
            )
            logger.info(f"Report generation complete: {artifact}")
            return artifact

        except FileNotFoundError as e:
            logger.error(f"Missing file: {e}")
            raise
        except PermissionError as e:
            logger.error(f"Permission denied writing report: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in ReportGenerator.run(): {e}")
            raise


# ── standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    from src.utils import load_dataframe
    from src.constants import PROCESSED_DATA_FILE

    df      = load_dataframe(str(PROCESSED_DATA_FILE))
    y_true  = df[TARGET_COLUMN].tolist()
    y_pred  = [0] * len(df)
    y_prob  = [0.3] * len(df)

    generator = ReportGenerator()
    artifact  = generator.run(df, y_true, y_pred, y_prob)

    print(f"\nReport saved to  : {artifact.report_path}")
    print(f"Charts saved to  : {artifact.charts_dir}")
    print(f"All checks passed: {artifact.all_bias_checks_passed}")
    print(f"Generated at     : {artifact.generated_at}")