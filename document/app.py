from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

# Support both `python document\app.py` and `python -m document.app`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from directplagiarism.detector import DirectPlagiarismDetector
from directplagiarism.extraction import extract_text_from_file, is_empty_or_weak_text
from directplagiarism.reporting import ReportGenerator


def allowed_file(filename: str, extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def load_saved_reports(report_dir: Path) -> list[dict]:
    reports: list[dict] = []
    for folder in report_dir.iterdir():
        report_file = folder / "analysis.json"
        if not folder.is_dir() or not report_file.is_file():
            continue
        try:
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        reports.append(
            {
                "report_id": report_data.get("report_id", folder.name),
                "submission_filename": report_data.get("submission_filename", "Untitled document"),
                "analysis_timestamp": report_data.get("analysis_timestamp", ""),
                "reference_count": len(report_data.get("reference_documents", [])),
                "overall_similarity": float(report_data.get("overall_similarity", 0.0)),
                "sort_time": folder.stat().st_mtime,
            }
        )

    return sorted(reports, key=lambda report: report["sort_time"], reverse=True)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "yourplagcheck-dev-key"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["ALLOWED_EXTENSIONS"] = {"pdf", "docx", "txt"}
    base_dir = Path(__file__).resolve().parent
    app.config["UPLOAD_FOLDER"] = base_dir / "uploads"
    app.config["REFERENCE_FOLDER"] = base_dir / "uploads" / "references"
    app.config["REPORT_FOLDER"] = base_dir / "reports"

    for folder in (app.config["UPLOAD_FOLDER"], app.config["REFERENCE_FOLDER"], app.config["REPORT_FOLDER"]):
        folder.mkdir(parents=True, exist_ok=True)

    detector = DirectPlagiarismDetector(min_passage_length=5)
    report_generator = ReportGenerator(base_dir)

    @app.route("/")
    @app.route("/dashboard")
    def dashboard():
        reports = load_saved_reports(report_generator.report_dir)
        report_count = len(reports)
        reference_count = sum(report["reference_count"] for report in reports)
        average_similarity = (
            sum(report["overall_similarity"] for report in reports) / report_count
            if report_count
            else 0.0
        )
        trend_reports = list(reversed([dict(report) for report in reports[:5]]))
        for report in reports:
            report.pop("sort_time")
        for report in trend_reports:
            report.pop("sort_time")
        return render_template(
            "index.html",
            recent_reports=reports[:5],
            report_count=report_count,
            reference_count=reference_count,
            average_similarity=average_similarity,
            trend_reports=trend_reports,
        )

    @app.route("/reports")
    def list_reports():
        reports = load_saved_reports(report_generator.report_dir)
        for report in reports:
            report.pop("sort_time")
        return render_template("reports.html", reports=reports)

    @app.route("/modules/<module>")
    def work_in_progress(module: str):
        modules = {
            "ai-detection": {
                "title": "AI writing detection",
                "eyebrow": "AI DETECTION",
                "description": "This module will assess writing patterns associated with AI-generated text. No AI-writing model is currently integrated, so this page does not analyze or classify documents.",
                "planned": ["Model selection and responsible-use guidance", "Evidence-based indicators with uncertainty", "Document-level analysis history"],
            },
            "cross-document": {
                "title": "Cross-document verification",
                "eyebrow": "CROSS-DOCUMENT VERIFICATION",
                "description": "This module is planned for comparing a collection of documents against one another. Current analysis only compares one submission with reference files you upload.",
                "planned": ["Batch document selection", "Pairwise overlap summary", "Shared passage review across documents"],
            },
            "documents": {
                "title": "Document library",
                "eyebrow": "DOCUMENTS",
                "description": "A managed document library is not available yet. Documents can currently be selected from your device when starting a direct-plagiarism analysis.",
                "planned": ["Browse and filter stored documents", "Document metadata and extraction status", "Safe retention and removal controls"],
            },
            "settings": {
                "title": "Workspace settings",
                "eyebrow": "SETTINGS",
                "description": "User-configurable analysis preferences are not available yet. Direct-plagiarism checks currently use the application’s built-in matching thresholds.",
                "planned": ["Configurable comparison thresholds", "Report display preferences", "Document retention controls"],
            },
            "profile": {
                "title": "Profile",
                "eyebrow": "PROFILE",
                "description": "This is a local, single-workspace application. User profiles and account management have not been implemented.",
                "planned": ["Local profile details", "Account preferences", "Workspace identity"],
            },
            "analytics": {
                "title": "Analytics and trends",
                "eyebrow": "ANALYTICS",
                "description": "Long-term trend analysis is not available yet. The dashboard shows a small summary of actual saved reports, without inferring trends from missing data.",
                "planned": ["Time-range filters", "Similarity trend summaries", "Exportable aggregate views"],
            },
        }
        module_details = modules.get(module)
        if module_details is None:
            abort(404)
        return render_template("work_in_progress.html", module=module_details)

    @app.route("/analyze", methods=["GET", "POST"])
    def analyze():
        if request.method == "POST":
            submission = request.files.get("document")
            reference_files = request.files.getlist("reference_documents")

            if not submission or not submission.filename:
                flash("Please select a document to analyze.", "error")
                return redirect(url_for("analyze"))

            if not reference_files or not any(file and file.filename for file in reference_files):
                flash("Please upload at least one reference document.", "error")
                return redirect(url_for("analyze"))

            if not allowed_file(submission.filename, app.config["ALLOWED_EXTENSIONS"]):
                flash("Submission file type is not supported. Use PDF, DOCX, or TXT.", "error")
                return redirect(url_for("analyze"))

            try:
                submission_name = secure_filename(submission.filename)
                submission_path = app.config["UPLOAD_FOLDER"] / submission_name
                submission.save(submission_path)
                submission_text, _ = extract_text_from_file(submission_path)

                if not submission_text.strip():
                    flash("No usable text could be extracted from the submission document.", "warning")
                    return redirect(url_for("analyze"))

                reference_documents: list[dict] = []
                reference_names: list[str] = []
                for reference in reference_files:
                    if not reference or not reference.filename:
                        continue
                    if not allowed_file(reference.filename, app.config["ALLOWED_EXTENSIONS"]):
                        flash(f"Reference file {reference.filename} is not supported and was skipped.", "warning")
                        continue
                    reference_name = secure_filename(reference.filename)
                    reference_path = app.config["REFERENCE_FOLDER"] / reference_name
                    reference.save(reference_path)
                    try:
                        reference_text, _ = extract_text_from_file(reference_path)
                    except ValueError as exc:
                        flash(f"Reference file {reference.filename} could not be processed: {exc}", "warning")
                        continue

                    if not reference_text.strip():
                        flash(f"Reference file {reference.filename} produced no usable text and was skipped.", "warning")
                        continue

                    reference_documents.append({"filename": reference.filename, "text": reference_text})
                    reference_names.append(reference.filename)

                if not reference_documents:
                    flash("No reference documents contained usable text for comparison.", "error")
                    return redirect(url_for("analyze"))

                comparison = detector.compare_documents(submission_text, reference_documents)
                report = report_generator.build_report(submission.filename, reference_names, comparison)
                return redirect(url_for("view_report", report_id=report["report_id"]))
            except Exception as exc:  # pragma: no cover - defensive guard
                flash(f"Analysis failed: {exc}", "error")
                return redirect(url_for("analyze"))

        return render_template("analyze.html")

    @app.route("/reports/<report_id>")
    def view_report(report_id: str):
        if not re.fullmatch(r"plagcheck-[0-9]{20}Z", report_id):
            abort(404)
        report_dir = report_generator.report_dir / report_id
        if not report_dir.exists():
            flash("The requested report does not exist.", "error")
            return redirect(url_for("dashboard"))

        report_data = json.loads((report_dir / "analysis.json").read_text(encoding="utf-8"))
        return render_template("report.html", report=report_data)

    @app.route("/reports/<report_id>/download")
    def download_report(report_id: str):
        if not re.fullmatch(r"plagcheck-[0-9]{20}Z", report_id):
            abort(404)
        report_dir = report_generator.report_dir / report_id
        pdf_path = report_dir / "report.pdf"
        if not pdf_path.exists():
            flash("No PDF report is available for this analysis.", "error")
            return redirect(url_for("dashboard"))
        return send_file(pdf_path, as_attachment=True, download_name=f"{report_id}.pdf")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)