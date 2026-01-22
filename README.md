# PDF Resampler

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library and command-line tool to reduce PDF file size by resampling embedded images to a target DPI. Text and vector graphics are preserved without quality loss.

## Features

- 📉 **Reduce PDF file size** by resampling high-DPI images
- 🔤 **Preserve text and vectors** — only images are modified
- 🖼️ **Smart detection** — skips images already at or below target DPI
- 🎨 **Transparency support** — handles images with alpha channels
- 📦 **Dual interface** — use as a library or command-line tool
- ⚡ **Fast processing** — powered by pikepdf and Pillow

## Installation

### From Source

```bash
git clone https://github.com/yourusername/pdf-resampler.git
cd pdf-resampler
pip install -e .
```

### Dependencies Only

```bash
pip install pikepdf Pillow
```

## Local Development

### Setting Up Virtual Environment

1. **Clone the repository:**

```bash
git clone https://github.com/AlphaCloudTechnologies/py-resample-pdf.git
cd pdf-resampler
```

2. **Create a virtual environment:**

```bash
python3 -m venv venv
```

3. **Activate the virtual environment:**

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

5. **Install in editable mode (recommended for development):**

```bash
pip install -e .
```

### Running Locally Without Installing

You can run the tool directly without installing the package:

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Run as a module
python -m pdf_resampler.cli ./input/input4.pdf ./output/output4.pdf --dpi 150

# Or use the library in Python
python -c "from pdf_resampler import resample_pdf; resample_pdf('input.pdf', 'output.pdf')"
```

### Running After Installation

Once installed with `pip install -e .`, you can use the CLI command directly:

```bash
pdf-resampler input.pdf output.pdf --dpi 150
```

## Quick Start

### Command Line

```bash
# Basic usage (150 DPI default)
pdf-resampler input.pdf output.pdf

# Custom DPI
pdf-resampler input.pdf output.pdf --dpi 100

# Custom DPI and quality
pdf-resampler input.pdf output.pdf --dpi 120 --quality 80

# Maximum compression
pdf-resampler input.pdf output.pdf --dpi 72 --quality 60

# Silent mode
pdf-resampler input.pdf output.pdf --silent
```

### Python Library

```python
from pdf_resampler import resample_pdf

# Simple usage
result = resample_pdf("input.pdf", "output.pdf")
print(f"Reduced by {result.reduction_percent:.1f}%")

# With options
result = resample_pdf(
    "input.pdf",
    "output.pdf",
    dpi=100,
    quality=75,
    verbose=True
)

# Access detailed results
print(f"Images processed: {result.images_processed}")
print(f"Images skipped: {result.images_skipped}")
print(f"Original size: {result.original_size_kb:.1f} KB")
print(f"New size: {result.new_size_kb:.1f} KB")
print(f"Bytes saved: {result.bytes_saved}")
```

## API Reference

### `resample_pdf()`

Main function to resample images in a PDF.

```python
resample_pdf(
    input_path: str,
    output_path: str,
    dpi: int = 150,
    quality: int = 85,
    verbose: bool = False
) -> ResampleResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | — | Path to the input PDF file |
| `output_path` | str | — | Path for the output PDF file |
| `dpi` | int | 150 | Target DPI for images |
| `quality` | int | 85 | JPEG quality (1-100) |
| `verbose` | bool | False | Print progress information |

**Returns:** `ResampleResult` object

### `ResampleResult`

Result object with processing statistics.

| Attribute | Type | Description |
|-----------|------|-------------|
| `images_processed` | int | Number of images resampled |
| `images_skipped` | int | Number of images skipped |
| `original_size` | int | Original file size (bytes) |
| `new_size` | int | New file size (bytes) |
| `reduction_percent` | float | Percentage size reduction |
| `original_size_kb` | float | Original size in KB |
| `new_size_kb` | float | New size in KB |
| `bytes_saved` | int | Bytes saved |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `PDFResamplerError` | Base exception class |
| `MissingDependencyError` | Required package not installed |
| `FileProcessingError` | Error processing the PDF |
| `InvalidParameterError` | Invalid parameter provided |

## Command Line Options

```
usage: pdf-resampler [-h] [-d DPI] [-q 1-100] [-v] [-s] [--version] input output

positional arguments:
  input                 Input PDF file
  output                Output PDF file

options:
  -h, --help            Show help message and exit
  -d, --dpi DPI         Target DPI for images (default: 150)
  -q, --quality 1-100   JPEG quality 1-100 (default: 85)
  -v, --verbose         Show detailed progress
  -s, --silent          Suppress all output
  --version             Show version number
```

## Recommended DPI Values

| DPI | Use Case |
|-----|----------|
| 72 | Screen/web viewing only |
| 100 | Basic print quality |
| 150 | Good print quality (default) |
| 300 | High quality print |

## How It Works

1. **Scan** — Analyzes the PDF to find all embedded images, including those in Form XObjects
2. **Calculate** — Determines the effective DPI of each image based on its rendered size
3. **Filter** — Skips images already at or below the target DPI
4. **Resample** — Downscales qualifying images using Lanczos resampling
5. **Compress** — Encodes images as JPEG (or PNG for transparency)
6. **Save** — Writes the optimized PDF with maximum compression

## Requirements

- Python 3.8+
- pikepdf ≥ 8.0.0
- Pillow ≥ 9.0.0

## Disclaimer

This software is provided "as is", without warranty of any kind. **Use at your own risk.** The authors are not responsible for any damage or data loss that may occur from using this software. Always keep backups of your original files.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [pikepdf](https://github.com/pikepdf/pikepdf) — PDF library for Python
- [Pillow](https://github.com/python-pillow/Pillow) — Python Imaging Library

