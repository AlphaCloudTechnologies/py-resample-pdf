"""
Command-line interface for pdf_resampler.
"""

import argparse
import sys

from .core import resample_pdf
from .exceptions import PDFResamplerError


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="pdf-resampler",
        description="Resample images in a PDF to reduce file size while preserving text and vector graphics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.pdf output.pdf
  %(prog)s input.pdf output.pdf --dpi 100
  %(prog)s input.pdf output.pdf --dpi 150 --quality 75
  %(prog)s input.pdf output.pdf -d 72 -q 60  # Maximum compression

Recommended DPI values:
  72  - Screen/web viewing only
  100 - Basic print quality
  150 - Good print quality (default)
  300 - High quality print
        """
    )
    
    parser.add_argument(
        "input",
        help="Input PDF file"
    )
    parser.add_argument(
        "output",
        help="Output PDF file"
    )
    parser.add_argument(
        "-d", "--dpi", 
        type=int, 
        default=150,
        help="Target DPI for images (default: 150)"
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=85,
        choices=range(1, 101),
        metavar="1-100",
        help="JPEG quality 1-100 (default: 85)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "-s", "--silent",
        action="store_true",
        help="Suppress all output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    if args.silent:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = True  # Default to verbose for CLI
    
    try:
        result = resample_pdf(
            args.input,
            args.output,
            dpi=args.dpi,
            quality=args.quality,
            verbose=verbose
        )
        
        if not args.silent:
            if result.reduction_percent > 0:
                sys.exit(0)
            else:
                print("\nNote: File size did not decrease. Try lower DPI or quality settings.")
                sys.exit(0)
                
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PDFResamplerError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

