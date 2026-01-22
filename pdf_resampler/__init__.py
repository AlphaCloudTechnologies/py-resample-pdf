"""
PDF Resampler - Reduce PDF file size by resampling embedded images.

A Python library and CLI tool to reduce PDF file size by resampling/compressing
embedded images to a target DPI while preserving text and vector graphics.

Basic Usage:
    >>> from pdf_resampler import resample_pdf
    >>> result = resample_pdf("input.pdf", "output.pdf")
    >>> print(f"Reduced by {result.reduction_percent:.1f}%")

With Options:
    >>> result = resample_pdf(
    ...     "input.pdf",
    ...     "output.pdf",
    ...     dpi=100,
    ...     quality=75,
    ...     verbose=True
    ... )
"""

from .core import resample_pdf, ResampleResult
from .exceptions import (
    PDFResamplerError,
    MissingDependencyError,
    FileProcessingError,
    InvalidParameterError,
)

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    # Main function
    "resample_pdf",
    # Result class
    "ResampleResult",
    # Exceptions
    "PDFResamplerError",
    "MissingDependencyError",
    "FileProcessingError",
    "InvalidParameterError",
    # Metadata
    "__version__",
]

