"""Forensic Verification Report generation — PDF and JSON, per the architecture's
"Output: Forensic Verification Report" stage (match score, confidence interval,
heatmap, deviation summary).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from sigverify.pipeline.inference import VerificationResult


def to_json(result: VerificationResult, output_path: str | Path | None = None) -> dict:
    payload = result.to_json_safe()
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return payload


def _heatmap_to_image_buffer(heatmap: np.ndarray) -> io.BytesIO:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(heatmap, cmap="jet")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf


DECISION_COLORS = {"Genuine": colors.HexColor("#1a7f37"), "Review": colors.HexColor("#b8860b"), "Forged": colors.HexColor("#c0392b")}


def generate_pdf_report(result: VerificationResult, output_path: str | Path, reference_id: str = "N/A", query_id: str = "N/A") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, title="Signature Verification Report")
    story = []

    story.append(Paragraph("Forensic Signature Verification Report", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Reference ID: {reference_id} &nbsp;&nbsp;|&nbsp;&nbsp; Query ID: {query_id}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {result.timestamp}", styles["Normal"]))
    story.append(Spacer(1, 0.6 * cm))

    decision_style = styles["Heading2"].clone("decision")
    decision_style.textColor = DECISION_COLORS.get(result.decision, colors.black)
    story.append(Paragraph(f"Decision: {result.decision}", decision_style))
    story.append(Spacer(1, 0.3 * cm))

    ci_text = "N/A" if result.confidence_interval is None else f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}] (95%)"
    rows = [
        ["Combined decision score", f"{result.combined_score:.4f}"],
        ["Confidence interval", ci_text],
        ["Fused (static+dynamic) similarity", f"{result.fused_similarity:.4f}"],
        ["Static-image similarity", f"{result.static_similarity:.4f}"],
        ["Dynamic-stroke similarity", "N/A" if result.dynamic_similarity is None else f"{result.dynamic_similarity:.4f}"],
        ["Calibrated score", "N/A" if result.calibrated_score is None else f"{result.calibrated_score:.4f}"],
        ["Anomaly score (higher = more normal)", "N/A" if result.anomaly_score is None else f"{result.anomaly_score:.4f}"],
        ["Flagged as novel/out-of-distribution", "N/A" if result.is_novel is None else str(result.is_novel)],
        ["Static modality weight", f"{result.modality_weights['static_weight']:.3f}"],
        ["Dynamic modality weight", f"{result.modality_weights['dynamic_weight']:.3f}"],
    ]
    table = Table(rows, colWidths=[9 * cm, 7 * cm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.6 * cm))

    if result.shap_modality_split is not None:
        story.append(Paragraph("Modality contribution to decision (SHAP)", styles["Heading3"]))
        split_rows = [[k, f"{v:.1f}%"] for k, v in result.shap_modality_split.items()]
        split_table = Table(split_rows, colWidths=[9 * cm, 7 * cm])
        split_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        story.append(split_table)
        story.append(Spacer(1, 0.6 * cm))

    if result.static_heatmap is not None:
        story.append(Paragraph("Static branch Grad-CAM deviation heatmap", styles["Heading3"]))
        buf = _heatmap_to_image_buffer(result.static_heatmap)
        story.append(RLImage(buf, width=8 * cm, height=8 * cm))
        story.append(Spacer(1, 0.4 * cm))

    if result.dynamic_deviation_scores is not None:
        story.append(Paragraph("Dynamic branch: most deviant stroke timesteps", styles["Heading3"]))
        idx_text = ", ".join(str(i) for i in result.top_deviant_indices.tolist())
        story.append(Paragraph(f"Top deviant timestep indices (query sequence): {idx_text}", styles["Normal"]))

    doc.build(story)
    return output_path
