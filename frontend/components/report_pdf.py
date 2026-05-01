"""
Kubios-style PDF report generator.
Primary: reportlab.platypus (A4, 2-column layout)
Fallback: matplotlib PdfPages
"""
from io import BytesIO
from datetime import datetime
from typing import Optional
import numpy as np


def generate_pdf_report(
    report: dict,
    signal: Optional[np.ndarray] = None,
    fs: Optional[int] = None,
    peaks: Optional[np.ndarray] = None,
    signal_type: str = "ECG",
) -> bytes:
    """Generate a Kubios-style PDF report. Returns PDF bytes."""
    try:
        return _generate_with_reportlab(report, signal, fs, peaks, signal_type)
    except Exception as e:
        try:
            return _generate_with_matplotlib(report, signal_type)
        except Exception as e2:
            raise RuntimeError(f"PDF generation failed: {e} | {e2}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_symbol(risk_level: str) -> str:
    return {"normal": "v", "borderline": "!", "abnormal": "X", "unknown": "~"}.get(
        risk_level, "~"
    )


def _fmt_bm(bm: Optional[dict]) -> str:
    """Format a BiomarkerValue dict to string."""
    if bm is None or bm.get("value") is None:
        return "N/D"
    v = bm["value"]
    u = bm.get("unit", "")
    sym = _risk_symbol(bm.get("risk_level", "unknown"))
    return f"{v:.2f} {u} {sym}"


def _hex(color) -> str:
    """Return a 6-char hex string from a reportlab Color (strips '#')."""
    h = color.hexval()          # e.g. '#27ae60' or '0x27ae60'
    h = h.replace("#", "").replace("0x", "")
    return h[-6:]               # ensure exactly 6 chars


# ---------------------------------------------------------------------------
# reportlab generator
# ---------------------------------------------------------------------------

def _generate_with_reportlab(report, signal, fs, peaks, signal_type) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # --- Colour palette ---
    C_HEADER  = colors.HexColor("#2c3e50")
    C_NORMAL  = colors.HexColor("#27ae60")
    C_BORDER  = colors.HexColor("#f39c12")
    C_ABNORM  = colors.HexColor("#e74c3c")
    C_LIGHT   = colors.HexColor("#ecf0f1")
    C_SECTION = colors.HexColor("#3498db")

    COLOR_MAP = {
        "normal":    C_NORMAL,
        "borderline": C_BORDER,
        "abnormal":  C_ABNORM,
        "unknown":   colors.grey,
    }

    # --- Paragraph styles ---
    title_style = ParagraphStyle(
        "title", parent=styles["Normal"],
        fontSize=14, textColor=C_HEADER,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    sub_style = ParagraphStyle(
        "sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, alignment=TA_CENTER,
    )
    warn_style = ParagraphStyle(
        "warn", parent=styles["Normal"],
        fontSize=8, textColor=C_ABNORM, alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "cell", parent=styles["Normal"],
        fontSize=8.5, leading=13,
    )
    sec_hdr_style = ParagraphStyle(
        "sechdr", parent=styles["Normal"],
        fontSize=9, textColor=colors.white,
        fontName="Helvetica-Bold",
    )

    # --- Header block ---
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    dur = report.get("duration_seconds", 0) or 0
    n_peaks = report.get("n_peaks_detected", 0) or 0
    mean_hr_val = None
    td_sec = report.get("hrv_time") or {}
    if td_sec.get("mean_hr") and isinstance(td_sec["mean_hr"], dict):
        mean_hr_val = td_sec["mean_hr"].get("value")
    hr_str = f"{mean_hr_val:.0f} bpm" if mean_hr_val is not None else "N/D"

    story.append(Paragraph("ECG/PPG HRV ANALYSIS REPORT", title_style))
    story.append(Paragraph(f"Generated: {now_str}", sub_style))
    story.append(Paragraph(
        f"Signal: {signal_type}  |  Duration: {dur:.1f}s  |  Mean HR: {hr_str}  |  Beats: {n_peaks}",
        sub_style,
    ))
    story.append(Paragraph(
        "WARNING: FOR RESEARCH USE ONLY - NOT CE-MARKED MEDICAL DEVICE",
        warn_style,
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_SECTION))
    story.append(Spacer(1, 0.2 * cm))

    # --- Helper: extract biomarker rows for a section ---
    def bm_rows(section_key: str, field_keys: list) -> list:
        rows = []
        sec = report.get(section_key) or {}
        for k in field_keys:
            bm = sec.get(k)
            if bm and isinstance(bm, dict):
                val = bm.get("value")
                unit = bm.get("unit", "")
                rl = bm.get("risk_level", "unknown")
                sym = _risk_symbol(rl)
                val_str = f"{val:.2f}" if val is not None else "N/D"
                c = COLOR_MAP.get(rl, colors.grey)
                rows.append([
                    Paragraph(
                        f'<font size="8">{bm.get("name", k)}</font>',
                        cell_style,
                    ),
                    Paragraph(
                        f'<font size="9" color="#{_hex(c)}"><b>{val_str} {unit} {sym}</b></font>',
                        cell_style,
                    ),
                ])
        return rows

    # ---------------------------------------------------------------
    # Section 1: TIME DOMAIN + FREQUENCY DOMAIN (side by side)
    # ---------------------------------------------------------------
    td_rows = bm_rows(
        "hrv_time",
        ["mean_hr", "sdnn", "rmssd", "pnn50", "nn50", "hrvi", "tinn", "sdann", "sdnni"],
    )
    fd_rows = bm_rows(
        "hrv_freq",
        ["total_power", "lf_power", "hf_power", "lf_hf_ratio", "lfnu", "hfnu", "vlf_power"],
    )
    max_rows = max(len(td_rows), len(fd_rows), 1)
    while len(td_rows) < max_rows:
        td_rows.append([Paragraph("", cell_style), Paragraph("", cell_style)])
    while len(fd_rows) < max_rows:
        fd_rows.append([Paragraph("", cell_style), Paragraph("", cell_style)])

    combined = [[
        Paragraph("<b>TIME DOMAIN</b>", sec_hdr_style), "", "",
        Paragraph("<b>FREQUENCY DOMAIN</b>", sec_hdr_style), "", "",
    ]]
    for td, fd in zip(td_rows, fd_rows):
        combined.append([td[0], td[1], "", fd[0], fd[1], ""])

    page_w = A4[0] - 3 * cm
    col_w = [
        page_w * 0.22, page_w * 0.18, page_w * 0.04,
        page_w * 0.22, page_w * 0.18, page_w * 0.16,
    ]
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), C_SECTION),
        ("BACKGROUND", (3, 0), (4, 0), C_SECTION),
        ("BACKGROUND", (0, 1), (1, -1), C_LIGHT),
        ("BACKGROUND", (3, 1), (4, -1), colors.HexColor("#e8f4f8")),
        ("GRID", (0, 0), (1, -1), 0.3, colors.white),
        ("GRID", (3, 0), (4, -1), 0.3, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (1, -1), [C_LIGHT, colors.white]),
        ("ROWBACKGROUNDS", (3, 1), (4, -1), [colors.HexColor("#e8f4f8"), colors.white]),
    ])
    t = Table(combined, colWidths=col_w)
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    # ---------------------------------------------------------------
    # Section 2: NONLINEAR ANALYSIS
    # ---------------------------------------------------------------
    nl_rows = bm_rows(
        "hrv_nonlinear",
        ["sd1", "sd2", "sd1_sd2_ratio", "sample_entropy", "dfa_alpha1"],
    )
    adv_rows = bm_rows(
        "hrv_advanced",
        ["dfa_alpha2", "approximate_entropy", "fuzzy_entropy", "mse_slope", "lyapunov_exponent"],
    )
    all_nl = nl_rows + adv_rows
    if all_nl:
        nl_header = [
            Paragraph("<b>NONLINEAR ANALYSIS</b>", sec_hdr_style),
            "", "", "", "",
        ]
        nl_data = [nl_header]
        for i in range(0, len(all_nl), 2):
            row = all_nl[i]
            next_row = all_nl[i + 1] if i + 1 < len(all_nl) else [
                Paragraph("", cell_style), Paragraph("", cell_style)
            ]
            nl_data.append([row[0], row[1], "", next_row[0], next_row[1]])

        nl_col_w = [
            page_w * 0.25, page_w * 0.18,
            page_w * 0.04,
            page_w * 0.25, page_w * 0.28,
        ]
        nl_ts = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_SECTION),
            ("SPAN", (0, 0), (-1, 0)),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
        nt = Table(nl_data, colWidths=nl_col_w)
        nt.setStyle(nl_ts)
        story.append(nt)
        story.append(Spacer(1, 0.3 * cm))

    # ---------------------------------------------------------------
    # Section 3: AUTONOMIC INDICES + ARRHYTHMIA (side by side)
    # ---------------------------------------------------------------
    auto_rows = bm_rows(
        "autonomic",
        ["pns_index", "sns_index", "baevsky_stress_index", "autonomic_balance"],
    )
    arr = report.get("arrhythmia") or {}
    ectopic_ratio = arr.get("ectopic_ratio", 0) or 0
    arr_rows = [
        [
            Paragraph("AFib Suspected", cell_style),
            Paragraph("YES !" if arr.get("afib_suspected") else "No v", cell_style),
        ],
        [
            Paragraph("HR Class", cell_style),
            Paragraph(str(arr.get("heart_rate_class", "N/D")).title(), cell_style),
        ],
        [
            Paragraph("Ectopic Beats", cell_style),
            Paragraph(
                f"{arr.get('ectopic_beats', 'N/D')} ({ectopic_ratio * 100:.1f}%)",
                cell_style,
            ),
        ],
        [
            Paragraph("RR CV", cell_style),
            Paragraph(f"{arr.get('rr_cv', 0) or 0:.4f}", cell_style),
        ],
    ]

    max_r = max(len(auto_rows), len(arr_rows), 1)
    while len(auto_rows) < max_r:
        auto_rows.append([Paragraph("", cell_style), Paragraph("", cell_style)])
    while len(arr_rows) < max_r:
        arr_rows.append([Paragraph("", cell_style), Paragraph("", cell_style)])

    aut_combined = [[
        Paragraph("<b>AUTONOMIC INDICES</b>", sec_hdr_style), "", "",
        Paragraph("<b>ARRHYTHMIA ANALYSIS</b>", sec_hdr_style), "", "",
    ]]
    for ar, arr_r in zip(auto_rows, arr_rows):
        aut_combined.append([ar[0], ar[1], "", arr_r[0], arr_r[1], ""])

    aut_col_w = [
        page_w * 0.22, page_w * 0.18, page_w * 0.04,
        page_w * 0.22, page_w * 0.34,
    ]
    # 5 logical columns but 6 data columns — drop last spacer column
    aut_col_w6 = [
        page_w * 0.22, page_w * 0.18, page_w * 0.04,
        page_w * 0.22, page_w * 0.20, page_w * 0.14,
    ]
    aut_ts = TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), C_SECTION),
        ("BACKGROUND", (3, 0), (4, 0), C_SECTION),
        ("BACKGROUND", (0, 1), (1, -1), colors.HexColor("#f0fff0")),
        ("BACKGROUND", (3, 1), (4, -1), colors.HexColor("#fff8f0")),
        ("ROWBACKGROUNDS", (0, 1), (1, -1), [colors.HexColor("#f0fff0"), colors.white]),
        ("ROWBACKGROUNDS", (3, 1), (4, -1), [colors.HexColor("#fff8f0"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    aut_t = Table(aut_combined, colWidths=aut_col_w6)
    aut_t.setStyle(aut_ts)
    story.append(aut_t)
    story.append(Spacer(1, 0.3 * cm))

    # ---------------------------------------------------------------
    # Footer: SIGNAL QUALITY
    # ---------------------------------------------------------------
    sqi = report.get("signal_quality") or {}
    score = sqi.get("overall_score")
    label = sqi.get("quality_label", "N/D")
    snr   = sqi.get("snr_db")
    flat  = sqi.get("flatline_fraction", 0) or 0
    clip  = sqi.get("clipping_fraction", 0) or 0

    if isinstance(score, (int, float)) and score is not None:
        if isinstance(snr, (int, float)) and snr is not None:
            sqi_text = (
                f"Signal Quality: {score:.0f}/100 [{label}]  |  "
                f"SNR: {snr:.1f} dB  |  Flatline: {flat * 100:.1f}%  |  "
                f"Clipping: {clip * 100:.1f}%"
            )
        else:
            sqi_text = f"Signal Quality: {score:.0f}/100 [{label}]"
    else:
        sqi_text = "Signal Quality: N/D"

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        sqi_text,
        ParagraphStyle(
            "sqi", parent=styles["Normal"],
            fontSize=8, textColor=colors.grey, alignment=TA_CENTER,
        ),
    ))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# matplotlib fallback
# ---------------------------------------------------------------------------

def _generate_with_matplotlib(report: dict, signal_type: str) -> bytes:
    """Fallback PDF via matplotlib (no reportlab required)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))  # A4
        fig.suptitle(
            f"ECG/PPG HRV Report — {signal_type}", fontsize=12, fontweight="bold"
        )

        ax = axes[0][0]
        ax.axis("off")
        ax.set_title("Time Domain", fontsize=9)
        td = report.get("hrv_time") or {}
        rows = [
            [k, f"{v['value']:.2f} {v['unit']}" if v and v.get("value") is not None else "N/D"]
            for k, v in td.items() if isinstance(v, dict)
        ]
        if rows:
            ax.table(
                cellText=rows[:8], colLabels=["Parameter", "Value"],
                loc="center", cellLoc="left",
            )

        ax = axes[0][1]
        ax.axis("off")
        ax.set_title("Frequency Domain", fontsize=9)
        fd = report.get("hrv_freq") or {}
        rows = [
            [k, f"{v['value']:.3f} {v['unit']}" if v and v.get("value") is not None else "N/D"]
            for k, v in fd.items() if isinstance(v, dict)
        ]
        if rows:
            ax.table(
                cellText=rows[:8], colLabels=["Parameter", "Value"],
                loc="center", cellLoc="left",
            )

        for ax in axes.flat[2:]:
            ax.axis("off")

        fig.text(
            0.5, 0.01,
            "FOR RESEARCH USE ONLY - NOT CE-MARKED",
            ha="center", fontsize=7, color="red",
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return buf.getvalue()
