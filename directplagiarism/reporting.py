from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


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
        payload = {
            "framework": "AI-Based Plagiarism Detection Framework",
            "version": "0.1",
            "report_id": report_id,
            "submission_filename": submission_filename,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_documents": reference_documents,
            "overall_similarity": detector_result.get("overall_similarity", 0.0),
            "score_breakdown": detector_result.get("score_breakdown", {}),
            "matching_passage_count": len(detector_result.get("matches", [])),
            "matches": detector_result.get("matches", []),
            "status": detector_result.get("status", "No qualifying matches detected"),
            "methodology": "Normalized text comparison, sentence-level exact/near-exact overlap, and TF-IDF cosine similarity are combined to assess potential direct copying without asserting guilt from a single score.",
            "limitations": [
                "Text extraction can fail for unsupported or corrupted documents.",
                "OCR for scanned PDFs is marked as work in progress unless explicitly implemented.",
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
        if not matches:
            summary_html = "<p>No qualifying matches were detected under the configured comparison rules.</p>"
        else:
            summary_html = "".join(
                [
                    "<div class='match-item'>"
                    f"<h3>{escape(match.get('reference_document', 'reference'))}</h3>"
                    f"<p><strong>Type:</strong> {escape(str(match.get('match_type', 'unknown')))} | "
                    f"<strong>Similarity:</strong> {match.get('similarity', 0.0)}</p>"
                    f"<p><strong>Submission excerpt:</strong> <mark>{escape(str(match.get('submission_excerpt', '')))}</mark></p>"
                    f"<p><strong>Reference excerpt:</strong> {escape(str(match.get('reference_excerpt', '')))}</p>"
                    "</div>"
                    for match in matches
                ]
            )

        return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Plagiarism Report {escape(str(payload.get('report_id', '')))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f7f9fc; color: #222; }}
    .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,.08); }}
    h1, h2 {{ color: #173b6c; }}
    .summary {{ background: #eef5ff; padding: 1rem; border-left: 6px solid #2d6cdf; margin-bottom: 1.5rem; }}
    .match-item {{ border: 1px solid #dfe7f3; padding: 1rem; margin-bottom: 1rem; border-radius: 8px; background: #fafcff; }}
    mark {{ background: #fff2a8; padding: 0.1rem 0.2rem; border-radius: 4px; }}
    code {{ background: #f2f4f7; padding: 0.2rem 0.4rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class='container'>
    <h1>{escape(str(payload.get('framework', 'Plagiarism Report')))} v{escape(str(payload.get('version', '0.1')))}</h1>
    <div class='summary'>
      <p><strong>Report ID:</strong> <code>{escape(str(payload.get('report_id', '')))}</code></p>
      <p><strong>Submission:</strong> {escape(str(payload.get('submission_filename', 'unknown')))}</p>
      <p><strong>Analysis timestamp:</strong> {escape(str(payload.get('analysis_timestamp', '')))}</p>
      <p><strong>Overall similarity:</strong> {payload.get('overall_similarity', 0.0)}</p>
      <p><strong>Status:</strong> {escape(str(payload.get('status', 'No qualifying matches detected')))}</p>
    </div>

    <h2>Methodology</h2>
    <p>{escape(str(payload.get('methodology', '')))}</p>

    <h2>Score Breakdown</h2>
    <ul>
      <li>Exact similarity: {payload.get('score_breakdown', {}).get('exact', 0.0)}</li>
      <li>Near-exact similarity: {payload.get('score_breakdown', {}).get('near_exact', 0.0)}</li>
      <li>TF-IDF cosine similarity: {payload.get('score_breakdown', {}).get('tfidf', 0.0)}</li>
    </ul>

    <h2>Reference documents compared</h2>
    <ul>
      {''.join(f'<li>{escape(ref)}</li>' for ref in payload.get('reference_documents', []))}
    </ul>

    <h2>Matching passages</h2>
    {summary_html}

    <h2>Limitations</h2>
    <ul>
      {''.join(f'<li>{escape(item)}</li>' for item in payload.get('limitations', []))}
    </ul>
  </div>
</body>
</html>"""

    def generate_pdf_report(self, payload: dict, output_path: str | os.PathLike[str]) -> str:
        output = Path(output_path)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("AI-Based Plagiarism Detection Framework v0.1", styles['Title']))
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph(f"Report ID: {payload.get('report_id', '')}", styles['Normal']))
        story.append(Paragraph(f"Submission: {payload.get('submission_filename', '')}", styles['Normal']))
        story.append(Paragraph(f"Analysis Timestamp: {payload.get('analysis_timestamp', '')}", styles['Normal']))
        story.append(Paragraph(f"Overall Similarity: {payload.get('overall_similarity', 0.0)}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Methodology", styles['Heading2']))
        story.append(Paragraph(payload.get('methodology', ''), styles['Normal']))

        story.append(Paragraph("Matching passages", styles['Heading2']))
        matches = payload.get('matches', [])
        if not matches:
            story.append(Paragraph("No qualifying matches were detected under the configured comparison rules.", styles['Normal']))
        else:
            for match in matches:
                story.append(Paragraph(
                    f"Reference: {match.get('reference_document', 'unknown')} | Type: {match.get('match_type', 'unknown')} | Similarity: {match.get('similarity', 0.0)}",
                    styles['Normal'],
                ))
                story.append(Paragraph(f"Submission excerpt: {match.get('submission_excerpt', '')}", styles['Normal']))
                story.append(Paragraph(f"Reference excerpt: {match.get('reference_excerpt', '')}", styles['Normal']))
                story.append(Spacer(1, 0.15 * inch))

        doc = SimpleDocTemplate(str(output), pagesize=letter, title="Plagiarism Report", author="YourPlagcheck")
        doc.build(story)
        return str(output)
