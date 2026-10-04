
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "creditcard.csv"
METRICS_FILE = PROJECT_DIR / "reports" / "model_comparison.csv"
OUTPUT_FILE = PROJECT_DIR / "reports" / "fraud_analysis_report.pdf"


def load_metrics():
    if not METRICS_FILE.exists():
        return []

    with open(
        METRICS_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        return list(reader)


def format_percentage(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_seconds(value):
    try:
        return f"{float(value):.2f} sec"
    except (ValueError, TypeError):
        return "N/A"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("Reading dataset...")
    df = pd.read_csv(DATA_FILE)

    if not {"Amount", "Class"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain Amount and Class columns."
        )

    total = len(df)
    fraud_count = int((df["Class"] == 1).sum())
    legitimate_count = int((df["Class"] == 0).sum())

    fraud_percentage = (
        fraud_count / total * 100 if total else 0
    )

    average_amount = (
        float(df["Amount"].mean()) if total else 0
    )

    metrics = load_metrics()

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#101c36"),
        fontSize=22,
        leading=28,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#146cce"),
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = styles["BodyText"]
    body_style.leading = 16

    document = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    story = []

    # Title
    story.append(Paragraph(
        "FraudGuard",
        title_style
    ))

    story.append(Paragraph(
        "AI-Powered Fraud Detection &amp; Risk Analytics Platform",
        ParagraphStyle(
            "Subtitle",
            parent=body_style,
            alignment=TA_CENTER,
            fontSize=12,
        )
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        body_style
    ))

    # Project overview
    story.append(Paragraph(
        "1. Project Overview",
        heading_style
    ))

    story.append(Paragraph(
        "FraudGuard is a machine-learning project designed to "
        "identify potentially fraudulent credit card transactions. "
        "It uses transaction data to train classification models, "
        "estimate fraud risk, and display results through a web "
        "dashboard. The application also stores prediction history "
        "and supports CSV report downloads.",
        body_style
    ))

    # Dataset summary
    story.append(Paragraph(
        "2. Dataset Summary",
        heading_style
    ))

    summary_data = [
        ["Metric", "Value"],
        ["Total transactions", f"{total:,}"],
        ["Legitimate transactions", f"{legitimate_count:,}"],
        ["Fraudulent transactions", f"{fraud_count:,}"],
        ["Fraud percentage", f"{fraud_percentage:.4f}%"],
        ["Average transaction amount", f"{average_amount:.2f}"],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[3.0 * inch, 2.5 * inch],
        repeatRows=1,
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#101c36")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 1), (-1, -1),
         colors.HexColor("#f3f6fb")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(summary_table)

    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "The dataset is imbalanced when fraud cases form a small "
        "fraction of all transactions. Therefore, accuracy alone "
        "is not sufficient to evaluate a fraud detection model.",
        body_style
    ))

    # Model comparison
    story.append(Paragraph(
        "3. Model Performance Comparison",
        heading_style
    ))

    if metrics:
        columns = list(metrics[0].keys())

        display_columns = [
            column for column in columns
            if column.lower() in {
                "model",
                "fraud precision",
                "fraud recall",
                "fraud f1",
                "pr-auc",
                "roc-auc",
                "training time (seconds)",
            }
        ]

        # Fall back to available columns if names differ.
        if not display_columns:
            display_columns = columns

        table_data = [
            [column.replace("_", " ").title()
             for column in display_columns]
        ]

        for metric in metrics:
            row = []

            for column in display_columns:
                value = metric.get(column, "")

                if column.lower() in {
                    "fraud precision",
                    "fraud recall",
                    "fraud f1",
                    "pr-auc",
                    "roc-auc",
                }:
                    value = format_percentage(value)

                elif column.lower() == "training time (seconds)":
                    value = format_seconds(value)

                row.append(str(value))

            table_data.append(row)

        model_table = Table(
            table_data,
            repeatRows=1,
            hAlign="LEFT",
        )

        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#146cce")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(model_table)

    else:
        story.append(Paragraph(
            "Model comparison results were not found. "
            "Run src/train_model.py to generate the metrics file.",
            body_style
        ))

    # Limitations
    story.append(Paragraph(
        "4. Interpretation and Limitations",
        heading_style
    ))

    story.append(Paragraph(
        "Model metrics depend on the dataset, train-test split, "
        "and decision threshold. The risk scores shown by the "
        "application should be treated as experimental estimates, "
        "not guarantees that a transaction is fraudulent. "
        "A production deployment would require further testing, "
        "threshold calibration, monitoring, and privacy safeguards.",
        body_style
    ))

    # Conclusion
    story.append(Paragraph(
        "5. Conclusion",
        heading_style
    ))

    story.append(Paragraph(
        "FraudGuard demonstrates a complete fraud detection workflow "
        "covering data analysis, preprocessing, model comparison, "
        "risk prediction, transaction history, and report generation. "
        "Further work could include threshold tuning, improved "
        "monitoring, and evaluation on additional datasets.",
        body_style
    ))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(
            A4[0] / 2,
            0.35 * inch,
            f"FraudGuard Project Report | Page {doc.page}"
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    print("\nPDF report generated successfully!")
    print(f"Saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
