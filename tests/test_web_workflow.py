import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

from document.app import create_app
from directplagiarism.reporting import ReportGenerator


class WebWorkflowTests(unittest.TestCase):
    def test_saved_legacy_report_still_renders_without_new_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "document.app.ReportGenerator",
                side_effect=lambda _base_dir: ReportGenerator(temporary_directory),
            ):
                app = create_app()

            report_id = "plagcheck-20260101000000000000Z"
            report_directory = Path(temporary_directory) / "reports" / report_id
            report_directory.mkdir(parents=True)
            legacy_report = {
                "report_id": report_id,
                "submission_filename": "legacy.txt",
                "analysis_timestamp": "2026-01-01T00:00:00+00:00",
                "reference_documents": ["source.txt"],
                "overall_similarity": 0.5,
                "score_breakdown": {"exact": 0.5, "near_exact": 0.0, "tfidf": 0.7},
                "per_reference": [],
                "matching_passage_count": 1,
                "matches": [{
                    "match_type": "exact",
                    "similarity": 1.0,
                    "length": 5,
                    "submission_index": 0,
                    "reference_index": 0,
                    "reference_document": "source.txt",
                    "submission_excerpt": "A faithful legacy excerpt.",
                    "reference_excerpt": "A faithful legacy excerpt.",
                }],
            }
            (report_directory / "analysis.json").write_text(json.dumps(legacy_report), encoding="utf-8")
            app.config["TESTING"] = True

            response = app.test_client().get(f"/reports/{report_id}")

            self.assertEqual(response.status_code, 200)
            self.assertIn(b"A faithful legacy excerpt.", response.data)

    def test_upload_generates_html_and_pdf_report_with_match_evidence(self):
        copied_passage = (
            "Researchers developed a reliable method for measuring urban air quality "
            "across different neighborhoods."
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_root = Path(temporary_directory) / "reports"
            with patch(
                "document.app.ReportGenerator",
                side_effect=lambda _base_dir: ReportGenerator(temporary_directory),
            ):
                app = create_app()

            app.config.update(
                TESTING=True,
                UPLOAD_FOLDER=Path(temporary_directory) / "uploads",
                REFERENCE_FOLDER=Path(temporary_directory) / "references",
            )
            app.config["UPLOAD_FOLDER"].mkdir()
            app.config["REFERENCE_FOLDER"].mkdir()

            with app.test_client() as client:
                response = client.post(
                    "/analyze",
                    data={
                        "document": (
                            io.BytesIO(f"Introductory notes. {copied_passage}".encode("utf-8")),
                            "submission.txt",
                        ),
                        "reference_documents": [
                            (io.BytesIO(copied_passage.encode("utf-8")), "source.txt")
                        ],
                    },
                    content_type="multipart/form-data",
                    follow_redirects=True,
                )

                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Matching passages", response.data)
                self.assertIn(copied_passage.encode("utf-8"), response.data)

                report_files = list(report_root.glob("*/analysis.json"))
                self.assertEqual(len(report_files), 1)
                report = json.loads(report_files[0].read_text(encoding="utf-8"))
                self.assertEqual(report["matching_passage_count"], 1)
                self.assertEqual(report["matches"][0]["reference_document"], "source.txt")
                self.assertEqual(report["submission_word_count"], 15)
                self.assertGreater(report["metrics"]["matched_words_estimate"], 0)
                self.assertIn("Signal breakdown", response.get_data(as_text=True))

                report_text = report_files[0].with_name("report.html").read_text(encoding="utf-8")
                self.assertIn("Similarity by reference", report_text)
                self.assertIn("Estimated matched words", report_text)
                pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(report_files[0].with_name("report.pdf")).pages)
                self.assertIn("Signal breakdown", pdf_text)
                self.assertIn("Passage-level evidence", pdf_text)
                self.assertIn("Submission:", pdf_text)

                download = client.get(f"/reports/{report['report_id']}/download")
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.mimetype, "application/pdf")
                self.assertTrue(download.data.startswith(b"%PDF"))
                download.close()


if __name__ == "__main__":
    unittest.main()
