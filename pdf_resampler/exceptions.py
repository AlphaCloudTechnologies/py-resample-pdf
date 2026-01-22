"""
Custom exceptions for pdf_resampler.
"""


class PDFResamplerError(Exception):
    """Base exception for pdf_resampler errors."""
    pass


class MissingDependencyError(PDFResamplerError):
    """Raised when a required dependency is not installed."""
    pass


class FileProcessingError(PDFResamplerError):
    """Raised when there is an error processing a PDF file."""
    pass


class InvalidParameterError(PDFResamplerError):
    """Raised when an invalid parameter is provided."""
    pass

