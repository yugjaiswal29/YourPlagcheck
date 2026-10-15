"""Direct plagiarism detection package."""

from .detector import DirectPlagiarismDetector
from .extraction import extract_text_from_file
from .reporting import ReportGenerator

__all__ = ["DirectPlagiarismDetector", "extract_text_from_file", "ReportGenerator"]
