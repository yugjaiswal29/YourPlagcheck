from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#263b62")
BLUE = colors.HexColor("#496da8")
PERIWINKLE = colors.HexColor("#aab5d2")
GOLD = colors.HexColor("#f2bd3e")
RED = colors.HexColor("#a94452")
INK = colors.HexColor("#27364d")
MUTED = colors.HexColor("#66748a")
PALE = colors.HexColor("#f2f5fa")
RULE = colors.HexColor("#dce3ee")
WHITE = colors.white


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ratio(value: Any) -> float:
    return max(0.0, min(1.0, _number(value)))


def _percent(value: Any) -> str:
    return f"{_ratio(value) * 100:.1f}%"


def _safe_text(value: Any) -> str:
    return escape(str(value or ""), quote=False)


class HorizontalBarChart(Flowable):
    def __init__(self, items: list[tuple[str, float, colors.Color]], row_height: float = 25):
        super().__init__()
        self.items = items
        self.row_height = row_height

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = available_width
        self.height = max(self.row_height, len(self.items) * self.row_height)
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        label_width = min(175, self.width * 0.34)
        value_width = 45
        bar_x = label_width + 10
        bar_width = max(40, self.width - bar_x - value_width - 8)
        for index, (label, value, color) in enumerate(self.items):
            y = self.height - (index + 1) * self.row_height + 7
            display_label = label
            while display_label and stringWidth(display_label, "Helvetica", 8) > label_width:
                display_label = display_label[:-4] + "..." if len(display_label) > 4 else ""
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(INK)
            canvas.drawString(0, y + 2, display_label)
            canvas.setFillColor(PALE)
            canvas.roundRect(bar_x, y, bar_width, 9, 4, fill=1, stroke=0)
            canvas.setFillColor(color)
            canvas.roundRect(bar_x, y, bar_width * _ratio(value), 9, 4, fill=1, stroke=0)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(self.width, y + 1, _percent(value))


class ReportGenerator:
    def __init__(self, base_dir: str | os.PathLike[str]):
        self.base_dir = Path(base_dir)
        self.report_dir = self.base_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _make_report_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%fZ")
        return f"plagcheck-{timestamp}"

    def _ensure_safe_path(self, report_id: str) -> Path:
        report_dir = self.report_dir / report_id
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def build_report(self, submission_filename: str, reference_documents: list[str], detector_result: dict) -> dict:
        report_id = self._make_report_id()
        report_dir = self._ensure_safe_path(report_id)
        breakdown = detector_result.get("score_breakdown", {})
        word_count = max(0, int(detector_result.get("submission_word_count", 0)))
        exact_coverage = _ratio(breakdown.get("exact", 0.0))
        near_coverage = _ratio(breakdown.get("near_exact", 0.0))
        matches = detector_result.get("matches", [])
        payload = {
            "framework": "AI-Based Plagiarism Detection",
            "version": "0.1",
            "report_id": report_id,
            "submission_filename": submission_filename,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_documents": reference_documents,
            "submission_word_count": word_count,
            "overall_similarity": _ratio(detector_result.get("overall_similarity", 0.0)),
            "score_breakdown": breakdown,
            "per_reference": detector_result.get("per_reference", []),
            "matching_passage_count": len(matches),
            "exact_passage_count": sum(match.get("match_type") == "exact" for match in matches),
            "near_exact_passage_count": sum(match.get("match_type") == "near_exact" for match in matches),
            "metrics": {
                "exact_matched_words": round(word_count * exact_coverage),
                "near_exact_matched_words": round(word_count * near_coverage),
                "matched_words_estimate": round(word_count * min(1.0, exact_coverage + near_coverage)),
            },
            "matches": matches,
            "status": detector_result.get("status", "No qualifying matches detected"),
            "methodology": "The overall score is the highest direct-overlap coverage against any one reference. Exact passages are matched in order after Unicode normalization; near-exact passages may bridge small insertions or deletions only when the ordered sequence remains sufficiently similar. TF-IDF cosine similarity is shown as a separate contextual signal and does not increase the direct-overlap score. These signals are not proof of plagiarism.",
            "limitations": [
                "Text extraction can fail for unsupported or corrupted documents.",
                "Scanned PDFs require OCR, which is not currently implemented.",
                "Some phrase reuse may be legitimate common wording, so the engine reports evidence rather than definitive plagiarism.",
            ],
            "known_detection_status": "Document comparison completed using configured thresholds. Results indicate evidence of direct overlap only when qualifying passage matches are present.",
        }

        json_path = report_dir / "analysis.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        html_path = report_dir / "report.html"
        html_path.write_text(self.render_html_report(payload), encoding="utf-8")

        pdf_path = report_dir / "report.pdf"
        self.generate_pdf_report(payload, pdf_path)

        return {
            "report_id": report_id,
            "report_dir": str(report_dir),
            "json_path": str(json_path),
            "html_path": str(html_path),
            "pdf_path": str(pdf_path),
            "download_url": f"/reports/{report_id}/download",
            "report_url": f"/reports/{report_id}",
        }

    def render_html_report(self, payload: dict) -> str:
        matches = payload.get("matches", [])
        if matches:
            evidence_html = "".join(
                f"""<article class="evidence">
  <div class="evidence-head"><span class="tag">{_safe_text(match.get('match_type', 'unknown')).replace('_', ' ').title()}</span>
  <span>{_percent(match.get('similarity', 0))} sequence similarity</span></div>
  <p class="source">Reference: <strong>{_safe_text(match.get('reference_document', 'reference'))}</strong>
  · {int(match.get('matched_word_count', match.get('length', 0)))} matched words</p>
  <div class="excerpt"><small>SUBMISSION</small><p>{_safe_text(match.get('submission_excerpt'))}</p></div>
  <div class="excerpt reference"><small>REFERENCE</small><p>{_safe_text(match.get('reference_excerpt'))}</p></div>
</article>"""
                for match in matches
            )
        else:
            evidence_html = '<p class="empty">No qualifying passages were detected. This does not rule out overlap with sources that were not provided.</p>'

        breakdown = payload.get("score_breakdown", {})
        score_rows = [
            ("Exact passage coverage", breakdown.get("exact", 0), "#496da8"),
            ("Near-exact passage coverage", breakdown.get("near_exact", 0), "#f2bd3e"),
            ("TF-IDF context signal (not in score)", breakdown.get("tfidf", 0), "#aab5d2"),
        ]
        score_bars = "".join(
            f"""<div class="bar-row"><div><span>{label}</span><strong>{_percent(value)}</strong></div>
<div class="track"><i style="width:{_ratio(value) * 100:.1f}%;background:{color}"></i></div></div>"""
            for label, value, color in score_rows
        )
        sources = payload.get("per_reference", [])
        source_chart = "".join(
            f"""<div class="bar-row"><div><span>{_safe_text(source.get('reference_document', 'reference'))}</span>
<strong>{_percent(source.get('overall_similarity', 0))}</strong></div><div class="track">
<i style="width:{_ratio(source.get('overall_similarity', 0)) * 100:.1f}%;background:#496da8"></i></div></div>"""
            for source in sources
        ) or '<p class="empty">No reference comparison data is available.</p>'
        metrics = payload.get("metrics", {})

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Similarity analysis · {_safe_text(payload.get('report_id'))}</title>
  <style>
    :root {{ color-scheme:light; --navy:#263b62; --blue:#496da8; --ink:#27364d; --muted:#66748a; --line:#dce3ee; --pale:#f2f5fa; --gold:#f2bd3e; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:#eef1f6; color:var(--ink); font:15px/1.55 Arial,sans-serif; }}
    .page {{ max-width:1040px; margin:36px auto; padding:0 20px; }} .sheet {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:40px; box-shadow:0 12px 36px #263b6214; }}
    .brand {{ color:var(--blue); font-size:12px; letter-spacing:.15em; font-weight:bold; text-transform:uppercase; }}
    h1 {{ color:var(--navy); font-size:32px; line-height:1.15; margin:8px 0 6px; }} .subtitle {{ color:var(--muted); margin:0 0 26px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px 24px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:14px 0; color:var(--muted); font-size:13px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:22px 0; }}
    .metric {{ border:1px solid var(--line); background:#fbfcfe; border-radius:12px; padding:16px; }}
    .metric small {{ display:block; color:var(--muted); font-size:11px; font-weight:bold; letter-spacing:.07em; text-transform:uppercase; }}
    .metric strong {{ display:block; color:var(--navy); font-size:26px; margin:6px 0 2px; }} .metric span {{ color:var(--muted); font-size:12px; }}
    .section {{ margin-top:28px; }} h2 {{ color:var(--navy); font-size:19px; margin:0 0 12px; }}
    .columns {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }} .panel {{ border:1px solid var(--line); border-radius:12px; padding:20px; }}
    .bar-row {{ margin:13px 0; }} .bar-row>div:first-child {{ display:flex; justify-content:space-between; gap:12px; font-size:13px; margin-bottom:6px; }}
    .bar-row strong {{ color:var(--navy); white-space:nowrap; }} .track {{ height:9px; overflow:hidden; background:var(--pale); border-radius:10px; }}
    .track i {{ display:block; height:100%; border-radius:10px; }} .evidence {{ border:1px solid var(--line); border-radius:12px; padding:18px; margin:12px 0; break-inside:avoid; }}
    .evidence-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; color:var(--muted); font-size:13px; }}
    .tag {{ background:#edf2fb; border-radius:20px; color:var(--navy); padding:5px 10px; font-weight:bold; }}
    .source {{ color:var(--muted); font-size:13px; }} .excerpt {{ border-left:3px solid var(--blue); background:#f7f9fc; padding:10px 13px; margin-top:10px; }}
    .excerpt.reference {{ border-left-color:var(--gold); }} .excerpt small {{ color:var(--muted); font-size:10px; font-weight:bold; letter-spacing:.1em; }} .excerpt p {{ margin:5px 0 0; }}
    .note {{ background:#fff8e3; border:1px solid #f0d98f; border-radius:10px; padding:14px 16px; color:#564a2c; }}
    .muted,.empty {{ color:var(--muted); }} footer {{ border-top:1px solid var(--line); margin-top:30px; padding-top:14px; color:var(--muted); font-size:12px; }}
    @media(max-width:720px) {{ .sheet {{ padding:24px 18px; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .columns {{ grid-template-columns:1fr; }} h1 {{ font-size:27px; }} }}
    @media print {{ body {{ background:#fff; }} .page {{ margin:0 auto; padding:0; }} .sheet {{ box-shadow:none; border:0; padding:20px; }} }}
  </style>
</head>
<body><main class="page"><article class="sheet">
  <div class="brand">AI-Based Plagiarism Detection · Analysis report</div>
  <h1>Document similarity analysis</h1>
  <p class="subtitle">Evidence-led comparison summary for human review</p>
  <div class="meta"><span><strong>Submission:</strong> {_safe_text(payload.get('submission_filename', 'unknown'))}</span>
    <span><strong>Report ID:</strong> {_safe_text(payload.get('report_id'))}</span>
    <span><strong>Generated:</strong> {_safe_text(payload.get('analysis_timestamp'))}</span></div>
  <section class="metrics" aria-label="Analysis metrics">
    <div class="metric"><small>Highest direct overlap</small><strong>{_percent(payload.get('overall_similarity', 0))}</strong><span>Coverage against the best single reference</span></div>
    <div class="metric"><small>Passages flagged</small><strong>{int(payload.get('matching_passage_count', 0))}</strong><span>{int(payload.get('exact_passage_count', 0))} exact · {int(payload.get('near_exact_passage_count', 0))} near-exact</span></div>
    <div class="metric"><small>Estimated matched words</small><strong>{int(metrics.get('matched_words_estimate', 0)):,}</strong><span>Of {int(payload.get('submission_word_count', 0)):,} analyzed words</span></div>
    <div class="metric"><small>Sources compared</small><strong>{len(payload.get('reference_documents', []))}</strong><span>Uploaded reference documents</span></div>
  </section>
  <div class="note"><strong>Interpretation:</strong> { _safe_text(payload.get('status', '')) }. Similarity is a review signal, not a finding of plagiarism.</div>
  <section class="section columns">
    <div class="panel"><h2>Signal breakdown</h2>{score_bars}<p class="muted">Direct overlap drives the overall score. TF-IDF is contextual and is not added to it.</p></div>
    <div class="panel"><h2>Similarity by reference</h2>{source_chart}</div>
  </section>
  <section class="section"><h2>Passage-level evidence</h2>{evidence_html}</section>
  <section class="section"><h2>Method and limitations</h2><p>{_safe_text(payload.get('methodology', ''))}</p>
    <ul>{''.join(f'<li>{_safe_text(item)}</li>' for item in payload.get('limitations', []))}</ul>
  </section>
  <footer>Generated by {_safe_text(payload.get('framework'))}. Review flagged passages in their source and document context before drawing conclusions.</footer>
</article></main></body></html>"""

    def generate_pdf_report(self, payload: dict, output_path: str | os.PathLike[str]) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=23, leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5,
        ))
        styles.add(ParagraphStyle(
            name="ReportSubtitle", parent=styles["Normal"], fontSize=10,
            leading=14, textColor=MUTED, spaceAfter=14,
        ))
        styles.add(ParagraphStyle(
            name="ReportHeading", parent=styles["Heading2"], fontSize=13,
            leading=16, textColor=NAVY, spaceBefore=12, spaceAfter=7,
        ))
        styles.add(ParagraphStyle(
            name="ReportBody", parent=styles["BodyText"], fontSize=8.5,
            leading=12, textColor=INK, spaceAfter=5,
        ))
        styles.add(ParagraphStyle(
            name="ReportSmall", parent=styles["Normal"], fontSize=7.5,
            leading=10, textColor=MUTED,
        ))
        styles.add(ParagraphStyle(
            name="MetricValue", parent=styles["Normal"], fontName="Helvetica-Bold",
            fontSize=17, leading=20, textColor=NAVY, alignment=TA_LEFT,
        ))
        styles.add(ParagraphStyle(
            name="MetricLabel", parent=styles["Normal"], fontName="Helvetica-Bold",
            fontSize=6.5, leading=8, textColor=MUTED,
        ))

        def para(value: Any, style: str = "ReportBody") -> Paragraph:
            return Paragraph(_safe_text(value), styles[style])

        score = _ratio(payload.get("overall_similarity", 0))
        word_count = max(0, int(payload.get("submission_word_count", 0)))
        metrics = payload.get("metrics", {})
        source_count = len(payload.get("reference_documents", []))
        cards = [
            ("HIGHEST DIRECT OVERLAP", _percent(score)),
            ("PASSAGES FLAGGED", str(int(payload.get("matching_passage_count", 0)))),
            ("EST. MATCHED WORDS", f"{int(metrics.get('matched_words_estimate', 0)):,}"),
            ("SOURCES COMPARED", str(source_count)),
        ]
        card_width = letter[0] - 1.25 * inch
        cell_width = card_width / len(cards)
        card_table = Table(
            [
                [para(value, "MetricValue") for _, value in cards],
                [para(label, "MetricLabel") for label, _ in cards],
            ],
            colWidths=[cell_width] * len(cards),
            hAlign="LEFT",
        )
        card_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE),
            ("BOX", (0, 0), (-1, -1), 0.6, RULE),
            ("INNERGRID", (0, 0), (-1, -1), 0.6, WHITE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
            ("TOPPADDING", (0, 1), (-1, 1), 2),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
        ]))

        story: list[Any] = [
            para("Document similarity analysis", "ReportTitle"),
            para("Evidence-led comparison summary for human review", "ReportSubtitle"),
        ]
        metadata = [
            [para("<b>Submission</b>", "ReportSmall"), para(payload.get("submission_filename", "unknown"), "ReportSmall"),
             para("<b>Report ID</b>", "ReportSmall"), para(payload.get("report_id", ""), "ReportSmall")],
            [para("<b>Generated</b>", "ReportSmall"), para(payload.get("analysis_timestamp", ""), "ReportSmall"),
             para("<b>Words analyzed</b>", "ReportSmall"), para(f"{word_count:,}", "ReportSmall")],
        ]
        meta_table = Table(metadata, colWidths=[0.7*inch, 2.55*inch, 0.7*inch, 2.8*inch])
        meta_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ]))
        story.extend([meta_table, Spacer(1, 12), card_table, Spacer(1, 12)])

        interpretation = (
            f"<b>Interpretation:</b> {_safe_text(payload.get('status', ''))}. "
            "The score is a review signal based on direct overlap, not a finding of plagiarism."
        )
        callout = Table([[Paragraph(interpretation, styles["ReportBody"])]], colWidths=[card_width])
        callout.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e3")),
            ("BOX", (0, 0), (-1, -1), 0.7, GOLD),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([callout, para("Signal breakdown", "ReportHeading")])
        breakdown = payload.get("score_breakdown", {})
        story.append(HorizontalBarChart([
            ("Exact passage coverage", _ratio(breakdown.get("exact", 0)), BLUE),
            ("Near-exact passage coverage", _ratio(breakdown.get("near_exact", 0)), GOLD),
            ("TF-IDF context signal", _ratio(breakdown.get("tfidf", 0)), PERIWINKLE),
        ]))
        story.append(para(
            "Exact and near-exact direct overlap contribute to the reported score. "
            "TF-IDF is a separate vocabulary-context measure and does not increase that score.",
            "ReportSmall",
        ))

        story.append(para("Similarity by reference", "ReportHeading"))
        source_items = sorted(
            payload.get("per_reference", []),
            key=lambda item: _ratio(item.get("overall_similarity", 0)),
            reverse=True,
        )
        if source_items:
            visible_sources = source_items[:10]
            story.append(HorizontalBarChart([
                (
                    str(item.get("reference_document", "reference")),
                    _ratio(item.get("overall_similarity", 0)),
                    BLUE,
                )
                for item in visible_sources
            ], row_height=23))
            if len(source_items) > len(visible_sources):
                story.append(para(f"Showing the 10 highest-scoring references of {len(source_items)}.", "ReportSmall"))
        else:
            story.append(para("No reference comparison data is available.", "ReportBody"))

        story.append(para("Passage-level evidence", "ReportHeading"))
        matches = payload.get("matches", [])
        if not matches:
            story.append(para(
                "No qualifying passages were detected. This does not rule out overlap with sources that were not provided.",
                "ReportBody",
            ))
        else:
            for index, match in enumerate(matches, start=1):
                evidence = [
                    Paragraph(
                        f"<b>{index}. {_safe_text(match.get('match_type', 'unknown')).replace('_', ' ').title()} match</b>"
                        f" &nbsp;·&nbsp; {_percent(match.get('similarity', 0))} sequence similarity"
                        f" &nbsp;·&nbsp; {_safe_text(match.get('reference_document', 'reference'))}",
                        styles["ReportBody"],
                    ),
                    Paragraph(f"<b>Submission:</b> {_safe_text(match.get('submission_excerpt', ''))}", styles["ReportBody"]),
                    Paragraph(f"<b>Reference:</b> {_safe_text(match.get('reference_excerpt', ''))}", styles["ReportBody"]),
                ]
                story.append(KeepTogether(evidence))
                story.append(Spacer(1, 5))

        story.extend([
            para("Methodology and limitations", "ReportHeading"),
            para(payload.get("methodology", ""), "ReportBody"),
        ])
        for limitation in payload.get("limitations", []):
            story.append(para(f"• {_safe_text(limitation)}", "ReportSmall"))

        def draw_page(canvas: Any, document: Any) -> None:
            canvas.saveState()
            page_width, page_height = letter
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.5)
            canvas.line(document.leftMargin, page_height - 0.42 * inch, page_width - document.rightMargin, page_height - 0.42 * inch)
            canvas.setFont("Helvetica-Bold", 7)
            canvas.setFillColor(BLUE)
            canvas.drawString(document.leftMargin, page_height - 0.32 * inch, "AI-BASED PLAGIARISM DETECTION")
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(page_width - document.rightMargin, 0.38 * inch, f"Confidential analysis report  ·  Page {document.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(
            str(output),
            pagesize=letter,
            title="AI-Based Plagiarism Detection — Similarity Analysis",
            author="AI-Based Plagiarism Detection",
            subject="Direct-overlap analysis and passage-level evidence",
            leftMargin=0.65 * inch,
            rightMargin=0.65 * inch,
            topMargin=0.62 * inch,
            bottomMargin=0.62 * inch,
        )
        doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
        return str(output)
