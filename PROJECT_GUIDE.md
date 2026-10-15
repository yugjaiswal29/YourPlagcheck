# Project Guide: AI-Based Plagiarism Detection

This guide explains how the current application works, what the project files do, and why the implementation is structured this way. It is written for someone learning to explore and maintain the project.

## 1. What the application currently does

The working application is a local Flask website. A user uploads one submission and one or more reference files. The server extracts text, compares the submission against each reference, saves the results, then displays an analysis report and makes a PDF downloadable.

The current direct-comparison engine is deterministic. It looks for ordered passages of shared words and measures broad vocabulary similarity separately. Despite the product name, the current direct-plagiarism path does **not** use an AI language model, does not decide whether plagiarism occurred, and does not search the internet or a commercial academic database.

Supported uploads are PDF, DOCX, and TXT, up to the configured request-size limit. Scanned PDFs need OCR, which is not implemented. AI-writing detection, cross-document verification, settings, profiles, and a managed document library are marked as work in progress.

## 2. The application flow

```text
Browser
  |
  | GET pages / POST uploaded files
  v
document/app.py (Flask routes and request handling)
  |
  +--> directplagiarism/extraction.py
  |      Extract text from PDF, DOCX, or TXT and normalize whitespace
  |
  +--> directplagiarism/detector.py
  |      Compare normalized word sequences and calculate separate signals
  |
  +--> directplagiarism/reporting.py
  |      Save analysis.json, report.html, and report.pdf
  |
  +--> document/templates/
         Render dashboard, upload form, report, history, and WIP screens
```

### One analysis, step by step

1. The browser shows the upload form from `document/templates/analyze.html`.
2. The form posts a submission and reference files to the `/analyze` route in `document/app.py`.
3. Flask checks for the submission and references, accepts PDF/DOCX/TXT, applies the upload-size limit, and saves the files under `document/uploads/`.
4. `directplagiarism/extraction.py` reads the file text and normalizes whitespace.
5. `directplagiarism/detector.py` compares the submission separately with each reference. Matching passage evidence is kept separate from the TF-IDF context signal.
6. `directplagiarism/reporting.py` saves a JSON record and creates HTML and PDF versions of the report in a new folder under `document/reports/`.
7. Flask redirects the browser to `/reports/<report_id>`. The report page is rendered from `document/templates/report.html`; the PDF is available at `/reports/<report_id>/download`.

## 3. File-by-file guide

### Project root

| File/folder | Purpose |
|---|---|
| `README.md` | Short setup and run instructions for the current application. |
| `requirements.txt` | Python packages needed by the application: Flask, scikit-learn, pypdf, python-docx, and reportlab. |
| `venv/` | Local Python virtual environment used to run the project. It contains installed packages and is machine-specific, not application source. |
| `directplagiarism/` | Python package containing the extraction, comparison, and report-generation logic. |
| `document/` | Flask web application, page templates, static assets, uploaded files, and saved reports. |
| `tests/` | Regression tests for the comparison engine and web/report workflow. |
| `Frontend/` | Older prototype area. Its `index.html`, `scripts.js`, and `style.css` are empty; it is not used by the running Flask app. `wireframe.txt` records an earlier feature concept. |
| `Untitled Folder/` | Experimental preprocessing notebook and related notebook artifacts. They are not imported by the Flask app. The notebook has an old attempt that imports PyMuPDF (`fitz`) and other tools; the current app uses `pypdf` instead. |

### `directplagiarism/`

| File | Purpose and design |
|---|---|
| `__init__.py` | Marks the folder as a Python package and exposes the main detector, extraction function, and report generator for convenient imports. |
| `extraction.py` | Reads TXT, PDF, and DOCX files, normalizes whitespace, and provides small text helpers. It keeps file-format handling out of the comparison algorithm. |
| `detector.py` | Core direct-comparison logic. It normalizes tokens while retaining original text offsets, aligns token sequences in order, builds exact/near-exact passage evidence, counts covered submission words, and calculates TF-IDF as a separate context signal. |
| `reporting.py` | Converts detector output into a saved JSON record and polished HTML and PDF reports. It owns report formatting, summary metrics, charts, and report IDs rather than mixing those concerns into the Flask route. |

### `document/`

| File/folder | Purpose |
|---|---|
| `__init__.py` | Marks the Flask app folder as a Python package. |
| `app.py` | Flask application factory and routes. It validates upload presence and file type, stores uploads, calls extraction and comparison, renders pages, and serves saved reports. It also holds the WIP module descriptions. |
| `templates/base.html` | Shared page shell: branding, navigation, flash messages, and page footer. Other HTML templates extend it so global layout stays consistent. |
| `templates/index.html` | Dashboard with real counts and recent reports loaded from saved JSON reports. |
| `templates/analyze.html` | Submission/reference upload form and supported-format guidance. |
| `templates/report.html` | In-app analysis report with summary metrics, visual bars, reference breakdown, and passage evidence. |
| `templates/reports.html` | History/list page for saved analyses. |
| `templates/work_in_progress.html` | Reusable honest placeholder page for modules that are not implemented yet. |
| `static/style.css` | Shared application layout, responsive styling, theme colors, report charts, and component appearance. |
| `static/app.js` | Small upload-form enhancement: displays selected filenames and supports drag-and-drop. Analysis itself remains on the server. |
| `static/application-logo.png` | Application logo displayed in the shared navigation shell. |
| `uploads/` | Runtime storage for uploaded submission/reference documents. This is data, not source code. |
| `reports/` | Runtime storage for generated analysis folders. Each generated report folder includes `analysis.json`, `report.html`, and `report.pdf`. These are user data and should not be casually deleted. |

### `tests/`

| File | What it protects |
|---|---|
| `test_detector.py` | Checks identical and partial copies, near-exact edits, reordered text, common vocabulary, stopword-only documents, Unicode normalization, multiple passages, and multiple references. |
| `test_web_workflow.py` | Posts files through Flask, checks generated report content and PDF download, and checks that older reports without newer metrics still render. |

## 4. Why the design is structured this way

### A small Flask application instead of a separate API and frontend

The UI and server are currently one local project. Flask routes render HTML templates directly, which keeps setup simple for a student project: one Python process serves both pages and analysis. A separate React/Vue application plus JSON API could offer a richer interactive experience, but it would add another build system, dependencies, deployment configuration, and API contracts before those are necessary.

The `Frontend/` folder is a historical prototype, not the frontend currently shown at `http://127.0.0.1:5000/`. The active pages are the Jinja templates under `document/templates/`.

### Separate extraction, comparison, and reporting

These are different jobs with different failure modes:

- Extraction depends on document formats and parsers.
- Comparison should operate on text and produce evidence/metrics.
- Reporting should present and save that result.

Keeping them in separate modules makes it possible to test the detector without running a browser, and to change PDF/report presentation without rewriting the matching algorithm. A larger production service might introduce more formal interfaces or separate services, but that would add complexity for the current scale.

### Direct passage matching plus a separate TF-IDF signal

The detector uses ordered word sequences for passage evidence because order matters: a bag/set-of-words comparison can treat heavily reordered text as copied text. Original character offsets are retained so the report can show excerpts from the uploaded documents instead of reconstructing phrases from tokens.

TF-IDF is included as a broad vocabulary-context measure, but kept separate from the direct-overlap score. Common words can make unrelated documents look generally similar; adding that signal directly to a copying score would risk suggesting passage-level evidence that was never found.

This design intentionally favors clear evidence over claiming to detect every kind of reuse. More advanced semantic models could find paraphrased text, but they require model selection, additional dependencies/resources, careful thresholds, and a validated evaluation set. They can also return plausible but incorrect matches. They would be an extension, not a drop-in proof of plagiarism.

### Local files for storage instead of a database

Reports are saved as JSON plus rendered HTML/PDF in the filesystem. This is easy to inspect and suitable for a single-workspace prototype; no database server or schema migration is needed. The tradeoff is that it does not provide user accounts, concurrent multi-user isolation, robust retention/search, transactional updates, or centralized deployment storage. A multi-user deployment should move to a database and explicit storage/access-control policy.

### Explicit work-in-progress screens

Features such as AI-writing analysis and cross-document checking are visible in navigation but labeled as WIP. This avoids presenting a non-existent analysis as if it had run. A placeholder page is deliberately safer and clearer than a fake score or a success-shaped result.

## 5. How to explore and test the project

Run the application from the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe document\app.py
```

Then open `http://127.0.0.1:5000/`. Upload one submission and at least one reference from the Direct plagiarism page. New saved files will appear under `document/uploads/` and `document/reports/`.

Run all regression tests from the same root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

To follow the core logic, start with `document/app.py`, find the `/analyze` route, and follow its calls in order:

1. `extract_text_from_file(...)` in `directplagiarism/extraction.py`
2. `detector.compare_documents(...)` in `directplagiarism/detector.py`
3. `report_generator.build_report(...)` in `directplagiarism/reporting.py`
4. `view_report` and `document/templates/report.html`

## 6. Important caveats

- Only the reference files supplied to a run are compared. There is no web crawling or licensed academic-source index.
- This is direct text-overlap analysis, not a complete paraphrase, translation, citation, or AI-writing detector.
- A similarity score is not a plagiarism verdict; human review and source context are required.
- PDF/DOCX extraction may omit content or fail. Scanned PDFs do not receive OCR.
- Uploads and reports are stored locally. The app currently has no login or per-user access controls.
- `app.py` uses a development secret key and starts Flask with debug mode enabled. Do not expose the current development configuration directly to the public internet.

