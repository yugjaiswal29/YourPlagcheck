# AI-Based Plagiarism Detection

Version 0.1 is a local Flask application for comparing a submitted document with reference files provided by the user.

## Run locally

From the project folder in PowerShell:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe document\app.py
```

Open `http://127.0.0.1:5000/` in a browser. Upload a submission and one or more reference documents from the analysis page. PDF, DOCX, and TXT files are supported; scanned PDFs without selectable text are not OCR-processed.

The framework reports comparison signals and matching passages. Similarity scores do not determine plagiarism; review the evidence and source context before drawing conclusions.
