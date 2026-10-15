import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document.app import create_app
from directplagiarism.reporting import ReportGenerator


class WebWorkflowTests(unittest.TestCase):
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

                download = client.get(f"/reports/{report['report_id']}/download")
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.mimetype, "application/pdf")
                self.assertTrue(download.data.startswith(b"%PDF"))
                download.close()


if __name__ == "__main__":
    unittest.main()
