"""
PDF report generator using ReportLab.
"""

import os, io, datetime
from typing import Optional

def generate_report_pdf(record: dict, doctor_name: str = "Doctor") -> bytes:
    """Generate a PDF report for a prediction record. Returns raw PDF bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title", fontSize=20, textColor=colors.HexColor("#1a56db"),
                                     spaceAfter=4, fontName="Helvetica-Bold")
        heading_style = ParagraphStyle("Heading", fontSize=13, textColor=colors.HexColor("#1a56db"),
                                       spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
        body_style   = styles["Normal"]

        pred = record.get("prediction", {})
        inp  = record.get("patient_input", {})
        ts   = record.get("timestamp", "")
        try:
            dt = datetime.datetime.fromisoformat(ts).strftime("%d %b %Y, %H:%M UTC")
        except Exception:
            dt = ts

        CLASS_COLOR = {
            "Normal":      "#2ecc71",
            "Obstruction": "#e67e22",
            "Restriction": "#e74c3c",
        }
        predicted_class = pred.get("prediction", "Unknown")
        conf_pct = round(pred.get("confidence_pct", 0), 1)
        class_color = CLASS_COLOR.get(predicted_class, "#333333")

        story = []

        # ── Header ──────────────────────────────────────────────────────────────
        story.append(Paragraph("🫁 Lung Disease Diagnostic Report", title_style))
        story.append(Paragraph(f"Generated: {dt} &nbsp;&nbsp;|&nbsp;&nbsp; Physician: Dr. {doctor_name}", body_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccddff"), spaceAfter=10))

        # ── Prediction result ────────────────────────────────────────────────────
        story.append(Paragraph("Predicted Diagnosis", heading_style))
        result_data = [
            ["Diagnosis", "Confidence"],
            [predicted_class, f"{conf_pct}%"],
        ]
        result_table = Table(result_data, colWidths=[10*cm, 5*cm])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a56db")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor(class_color)),
            ("TEXTCOLOR",  (0, 1), (-1, 1), colors.white),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 13),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [None, None]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.white),
            ("ROUNDEDCORNERS", [4]),
        ]))
        story.append(result_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Probability distribution ─────────────────────────────────────────────
        story.append(Paragraph("Probability Distribution", heading_style))
        probs = pred.get("probabilities", {})
        prob_data = [["Class", "Probability"]]
        for cls, p in probs.items():
            prob_data.append([cls, f"{round(p * 100, 2)}%"])
        prob_table = Table(prob_data, colWidths=[8*cm, 7*cm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0fe")),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#ccddff")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8ff")]),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Explanation ──────────────────────────────────────────────────────────
        explanation = pred.get("explanation", {})
        text_summary = explanation.get("text_summary", "")
        top_features = explanation.get("top_features", [])

        if text_summary:
            story.append(Paragraph("Clinical Explanation", heading_style))
            story.append(Paragraph(text_summary, body_style))
            story.append(Spacer(1, 0.3*cm))

        if top_features:
            story.append(Paragraph("Top Contributing Features", heading_style))
            feat_data = [["Feature", "Contribution", "Direction"]]
            for f in top_features[:8]:
                feat_data.append([
                    f["feature"].replace("_", " "),
                    str(f["contribution"]),
                    "▲ Increases risk" if f["direction"] == "positive" else "▼ Decreases risk",
                ])
            feat_table = Table(feat_data, colWidths=[7*cm, 4*cm, 4*cm])
            feat_table.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#e8f0fe")),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#ccddff")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8ff")]),
            ]))
            story.append(feat_table)
            story.append(Spacer(1, 0.4*cm))

        # ── Patient inputs ───────────────────────────────────────────────────────
        story.append(Paragraph("Spirometry Input Values", heading_style))
        RAW_LABELS = {
            "Sex": "Sex (0=F, 1=M)", "Age": "Age (yrs)", "Weight": "Weight (kg)",
            "Height": "Height (cm)", "BMI": "BMI (kg/m²)",
            "Baseline_PEF_Ls": "PEF (L/s)", "Baseline_FEF2575_Ls": "FEF 25-75 (L/s)",
            "Baseline_Extrapolated_Volume": "Extrapolated Volume (L)",
            "Baseline_Forced_Expiratory_Time": "Forced Expiratory Time (s)",
            "Baseline_Number_Acceptable_Curves": "Acceptable Curves",
        }
        inp_data = [["Parameter", "Value"]]
        for key, label in RAW_LABELS.items():
            val = inp.get(key, "—")
            inp_data.append([label, str(val)])
        inp_table = Table(inp_data, colWidths=[9*cm, 6*cm])
        inp_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#e8f0fe")),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#ccddff")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8ff")]),
        ]))
        story.append(inp_table)

        # ── Footer ───────────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.8*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccddff")))
        story.append(Paragraph(
            "⚕ This report is generated by an AI-assisted diagnostic system and is intended "
            "to support, not replace, clinical judgement. Please consult a qualified pulmonologist "
            "for definitive diagnosis and treatment planning.",
            ParagraphStyle("Disclaimer", fontSize=8, textColor=colors.HexColor("#666666"),
                           spaceBefore=6)
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        # Minimal fallback if ReportLab not installed
        text = f"Lung Disease Report\n\nDiagnosis: {record.get('prediction', {}).get('prediction', 'N/A')}\n"
        return text.encode()
