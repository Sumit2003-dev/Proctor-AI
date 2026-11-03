"""Automated PDF & HTML Integrity Audit Report Generator for ProctorAI.

Generates an executive-ready post-exam audit report containing:
- Candidate session metadata and total monitored duration
- Overall integrity classification (CLEAN / WARNING / HIGH RISK)
- Incident breakdown & chronological violation log
- Photographic evidence thumbnails captured during violation events
- Official examiner verdict & sign-off block
"""

import datetime
import io
from pathlib import Path
from typing import List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import SNAPSHOTS_DIR


def generate_pdf_report(
    candidate_name: str,
    session_id: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    peak_score: float,
    status_label: str,
    alerts_df: pd.DataFrame,
    snapshot_paths: Optional[List[Path]] = None,
) -> bytes:
    """Generates a professional PDF integrity audit report in memory.

    Returns:
        Bytes of the generated PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    normal_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🛡️ PROCTORAI | OFFICIAL EXAM INTEGRITY AUDIT", title_style))
    story.append(
        Paragraph(
            f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Automated Integrity Assessment",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6")))
    story.append(Spacer(1, 12))

    # 2. Executive Summary Block
    duration = end_time - start_time
    duration_str = str(datetime.timedelta(seconds=int(duration.total_seconds())))

    # Color code badge
    verdict_color = (
        "#16a34a" if status_label == "NORMAL"
        else "#d97706" if status_label == "WARNING"
        else "#dc2626"
    )

    summary_data = [
        [
            Paragraph("<b>Candidate Name:</b>", normal_style),
            Paragraph(candidate_name, normal_style),
            Paragraph("<b>Session ID:</b>", normal_style),
            Paragraph(session_id, normal_style),
        ],
        [
            Paragraph("<b>Session Start:</b>", normal_style),
            Paragraph(start_time.strftime("%Y-%m-%d %H:%M:%S"), normal_style),
            Paragraph("<b>Session End:</b>", normal_style),
            Paragraph(end_time.strftime("%Y-%m-%d %H:%M:%S"), normal_style),
        ],
        [
            Paragraph("<b>Total Duration:</b>", normal_style),
            Paragraph(duration_str, normal_style),
            Paragraph("<b>Peak Suspicion Score:</b>", normal_style),
            Paragraph(f"<b>{peak_score:.1f} / 100</b>", normal_style),
        ],
        [
            Paragraph("<b>Integrity Status:</b>", normal_style),
            Paragraph(
                f"<font color='{verdict_color}'><b>{status_label}</b></font>",
                normal_style,
            ),
            Paragraph("<b>Total Incidents:</b>", normal_style),
            Paragraph(str(len(alerts_df)), normal_style),
        ],
    ]

    t_summary = Table(summary_data, colWidths=[120, 150, 120, 150])
    t_summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_summary)
    story.append(Spacer(1, 14))

    # 3. Incident History Log Table
    story.append(Paragraph("📋 Incident Timeline & Audit Trail", h2_style))
    if not alerts_df.empty:
        table_rows = [
            [
                Paragraph("<b>Timestamp</b>", normal_style),
                Paragraph("<b>Violation Type</b>", normal_style),
                Paragraph("<b>Score</b>", normal_style),
                Paragraph("<b>Details</b>", normal_style),
            ]
        ]
        # Include up to 25 most recent events to fit comfortably
        subset_df = alerts_df.tail(25)
        for _, row in subset_df.iterrows():
            table_rows.append(
                [
                    Paragraph(str(row.get("Timestamp", "")), normal_style),
                    Paragraph(str(row.get("Alert_Type", "")), normal_style),
                    Paragraph(str(row.get("Suspicion_Score", "")), normal_style),
                    Paragraph(str(row.get("Details", "")), normal_style),
                ]
            )

        t_alerts = Table(table_rows, colWidths=[110, 150, 60, 220])
        t_alerts.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t_alerts)
    else:
        story.append(
            Paragraph("<i>No integrity violations or suspicious behavior detected. Clean session.</i>", normal_style)
        )
    story.append(Spacer(1, 14))

    # 4. Photographic Evidence Section
    if snapshot_paths:
        story.append(Paragraph("📸 Photographic Evidence (Sample Frames)", h2_style))
        evidence_cells = []
        row_cells = []
        for path in snapshot_paths[:4]:  # Show up to 4 evidentiary snapshots
            try:
                img = Image(str(path), width=180, height=135)
                caption = Paragraph(f"<font size=7 color='#64748b'>{path.stem}</font>", normal_style)
                cell = [img, caption]
                row_cells.append(cell)
                if len(row_cells) == 2:
                    evidence_cells.append(row_cells)
                    row_cells = []
            except Exception:
                continue

        if row_cells:
            evidence_cells.append(row_cells)

        if evidence_cells:
            t_evidence = Table(evidence_cells, colWidths=[270, 270])
            t_evidence.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(t_evidence)
            story.append(Spacer(1, 12))

    # 5. Examiner Sign-off / Verdict Block
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#94a3b8")))
    story.append(Spacer(1, 8))

    verdict_text = (
        "PASSED (Clean Session)"
        if status_label == "NORMAL"
        else "NEEDS REVIEW (Flagged for Proctored Audit)"
        if status_label == "WARNING"
        else "SERIOUS BREACH SUSPECTED (Manual Investigation Required)"
    )

    verdict_data = [
        [
            Paragraph(f"<b>Final Recommendation:</b> {verdict_text}", normal_style),
            Paragraph("<b>Examiner Signature:</b> ___________________________", normal_style),
        ]
    ]
    t_verdict = Table(verdict_data, colWidths=[320, 220])
    t_verdict.setStyle(TableStyle([("PADDING", (0, 0), (-1, -1), 4)]))
    story.append(t_verdict)

    doc.build(story)
    return buffer.getvalue()
