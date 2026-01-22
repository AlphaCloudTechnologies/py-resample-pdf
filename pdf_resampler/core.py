"""
Core PDF resampling functionality.

This module provides the main logic for resampling images in PDF files.
"""

import io
import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .exceptions import MissingDependencyError, FileProcessingError, InvalidParameterError

try:
    import pikepdf
    from pikepdf import PdfImage
    from PIL import Image
except ImportError as e:
    raise MissingDependencyError(
        f"Missing required package: {e}. "
        "Install with: pip install pikepdf Pillow"
    )


@dataclass
class ResampleResult:
    """Result of a PDF resampling operation.
    
    Attributes:
        images_processed: Number of images that were resampled.
        images_skipped: Number of images skipped (already at or below target DPI).
        original_size: Original file size in bytes.
        new_size: New file size in bytes.
    """
    images_processed: int
    images_skipped: int
    original_size: int
    new_size: int
    
    @property
    def reduction_percent(self) -> float:
        """Percentage reduction in file size."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.new_size / self.original_size) * 100
    
    @property
    def original_size_kb(self) -> float:
        """Original file size in kilobytes."""
        return self.original_size / 1024
    
    @property
    def new_size_kb(self) -> float:
        """New file size in kilobytes."""
        return self.new_size / 1024
    
    @property
    def bytes_saved(self) -> int:
        """Number of bytes saved."""
        return self.original_size - self.new_size


def _multiply_matrix(m1: Tuple, m2: Tuple) -> Tuple:
    """Multiply two 3x3 transformation matrices (stored as 6 elements: a,b,c,d,e,f)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _get_matrix_scale(matrix: Tuple) -> Tuple[float, float]:
    """Get the scale factors from a transformation matrix."""
    a, b, c, d, e, f = matrix
    scale_x = math.sqrt(a * a + b * b)
    scale_y = math.sqrt(c * c + d * d)
    return scale_x, scale_y


def _parse_content_stream_for_xobjects(content_bytes: bytes, xobjects_dict) -> Dict:
    """
    Parse a PDF content stream and extract XObject usages with their transformation matrices.
    
    Returns:
        Dict mapping xobject_name -> list of (width_pts, height_pts, ctm)
    """
    xobject_usages = {}
    
    try:
        content = content_bytes.decode('latin-1')
    except Exception:
        return xobject_usages
    
    identity = (1, 0, 0, 1, 0, 0)
    ctm_stack = []
    current_ctm = identity
    
    token_pattern = re.compile(
        r'(\[.*?\])|'
        r'(\((?:[^()\\]|\\.)*\))|'
        r'(<[0-9A-Fa-f\s]*>)|'
        r'(/[^\s\[\]()<>/%]*)|'
        r'([+-]?(?:\d+\.?\d*|\.\d+))|'
        r'([A-Za-z*\'"]+)'
    , re.DOTALL)
    
    tokens = []
    for match in token_pattern.finditer(content):
        token = match.group(0).strip()
        if token:
            tokens.append(token)
    
    operand_stack = []
    
    for token in tokens:
        if token == 'q':
            ctm_stack.append(current_ctm)
        elif token == 'Q':
            if ctm_stack:
                current_ctm = ctm_stack.pop()
        elif token == 'cm':
            if len(operand_stack) >= 6:
                try:
                    f = float(operand_stack.pop())
                    e = float(operand_stack.pop())
                    d = float(operand_stack.pop())
                    c = float(operand_stack.pop())
                    b = float(operand_stack.pop())
                    a = float(operand_stack.pop())
                    new_matrix = (a, b, c, d, e, f)
                    current_ctm = _multiply_matrix(new_matrix, current_ctm)
                except (ValueError, IndexError):
                    pass
        elif token == 'Do':
            if operand_stack:
                name = operand_stack.pop()
                if name.startswith('/'):
                    scale_x, scale_y = _get_matrix_scale(current_ctm)
                    if name not in xobject_usages:
                        xobject_usages[name] = []
                    xobject_usages[name].append((scale_x, scale_y, current_ctm))
        elif token in ('BT', 'ET', 'BI', 'ID', 'EI', 'BDC', 'BMC', 'EMC', 
                       'MP', 'DP', 'sh', 'gs', 'CS', 'cs', 'SC', 'SCN',
                       'sc', 'scn', 'G', 'g', 'RG', 'rg', 'K', 'k', 'W', 'W*',
                       'n', 'f', 'F', 'f*', 'B', 'B*', 'b', 'b*', 's', 'S',
                       'm', 'l', 'c', 'v', 'y', 'h', 're', 'd', 'w', 'j', 'J',
                       'M', 'ri', 'i', 'Tc', 'Tw', 'Tz', 'TL', 'Tf', 'Tr',
                       'Ts', 'Td', 'TD', 'Tm', 'T*', 'Tj', 'TJ', "'", '"',
                       'd0', 'd1'):
            if token not in ('cm', 'Do', 'q', 'Q'):
                operand_stack.clear()
        else:
            operand_stack.append(token)
    
    return xobject_usages


def _collect_all_images(pdf) -> Dict:
    """
    Collect all images from the PDF, including those inside Form XObjects.
    
    Returns:
        Dict mapping objgen -> (name, obj, location, render_w_pts, render_h_pts)
    """
    all_images = {}
    
    def process_xobjects(xobjects, content_bytes, parent_ctm, location_prefix):
        """Process XObjects from a resources dictionary."""
        if not xobjects:
            return
        
        usages = _parse_content_stream_for_xobjects(content_bytes, xobjects)
        
        for name in list(xobjects.keys()):
            obj = xobjects[name]
            subtype = obj.get("/Subtype")
            name_str = str(name)
            if not name_str.startswith('/'):
                name_str = '/' + name_str
            
            usage_list = usages.get(name_str, [])
            
            if subtype == "/Image":
                objgen = obj.objgen
                
                max_w, max_h = 0, 0
                for scale_x, scale_y, ctm in usage_list:
                    combined = _multiply_matrix(ctm, parent_ctm) if parent_ctm != (1,0,0,1,0,0) else ctm
                    w, h = _get_matrix_scale(combined)
                    if w > max_w:
                        max_w = w
                    if h > max_h:
                        max_h = h
                
                if max_w == 0 or max_h == 0:
                    max_w, max_h = 612, 792
                
                if objgen not in all_images:
                    all_images[objgen] = (name, obj, location_prefix, max_w, max_h)
                else:
                    _, _, _, existing_w, existing_h = all_images[objgen]
                    if max_w > existing_w or max_h > existing_h:
                        all_images[objgen] = (name, obj, location_prefix, 
                                              max(max_w, existing_w), max(max_h, existing_h))
            
            elif subtype == "/Form":
                form_resources = obj.get("/Resources")
                if form_resources:
                    form_xobjects = form_resources.get("/XObject")
                    if form_xobjects:
                        try:
                            form_content = obj.read_bytes()
                        except Exception:
                            form_content = b""
                        
                        for scale_x, scale_y, ctm in usage_list:
                            combined_ctm = _multiply_matrix(ctm, parent_ctm) if parent_ctm != (1,0,0,1,0,0) else ctm
                            process_xobjects(
                                form_xobjects, 
                                form_content, 
                                combined_ctm,
                                f"{location_prefix} > Form {name}"
                            )
    
    for page_num, page in enumerate(pdf.pages, 1):
        resources = page.get("/Resources")
        if not resources:
            continue
        
        xobjects = resources.get("/XObject")
        if not xobjects:
            continue
        
        contents = page.get("/Contents")
        if contents:
            if isinstance(contents, pikepdf.Array):
                content_bytes = b""
                for stream in contents:
                    content_bytes += stream.read_bytes()
            else:
                content_bytes = contents.read_bytes()
        else:
            content_bytes = b""
        
        process_xobjects(xobjects, content_bytes, (1, 0, 0, 1, 0, 0), f"Page {page_num}")
    
    return all_images


def _update_all_image_references(pdf, image_replacements: Dict) -> None:
    """Update all references to replaced images, including in Form XObjects."""
    
    def update_xobjects(xobjects):
        """Update image references in an XObject dictionary."""
        if not xobjects:
            return
        
        for name in list(xobjects.keys()):
            obj = xobjects[name]
            subtype = obj.get("/Subtype")
            
            if subtype == "/Image":
                objgen = obj.objgen
                if objgen in image_replacements:
                    xobjects[name] = image_replacements[objgen]
            
            elif subtype == "/Form":
                form_resources = obj.get("/Resources")
                if form_resources:
                    form_xobjects = form_resources.get("/XObject")
                    update_xobjects(form_xobjects)
    
    for page in pdf.pages:
        resources = page.get("/Resources")
        if resources:
            xobjects = resources.get("/XObject")
            update_xobjects(xobjects)


def _extract_image_with_pdfimage(pdf_image, image_obj) -> Optional[Image.Image]:
    """Extract a PIL Image using pikepdf's PdfImage class.
    
    Handles SMask (soft mask/transparency) by combining it with the main image.
    """
    pil_image = None
    
    try:
        pil_image = pdf_image.as_pil_image()
    except Exception:
        try:
            raw_image = pdf_image.extract_to(io.BytesIO())
            raw_image.seek(0)
            pil_image = Image.open(raw_image)
        except Exception:
            try:
                data = pdf_image.read_bytes()
                pil_image = Image.open(io.BytesIO(data))
            except Exception:
                return None
    
    if pil_image is None:
        return None
    
    smask = image_obj.get("/SMask")
    if smask is not None:
        try:
            smask_image = PdfImage(smask)
            alpha_pil = smask_image.as_pil_image()
            
            if alpha_pil.mode != "L":
                alpha_pil = alpha_pil.convert("L")
            
            if alpha_pil.size != pil_image.size:
                alpha_pil = alpha_pil.resize(pil_image.size, Image.Resampling.LANCZOS)
            
            if pil_image.mode == "L":
                pil_image = pil_image.convert("LA")
                pil_image.putalpha(alpha_pil)
            else:
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                pil_image = pil_image.convert("RGBA")
                pil_image.putalpha(alpha_pil)
        except Exception:
            pass
    
    return pil_image


def _resample_image(
    img: Image.Image,
    target_dpi: int,
    current_dpi: float,
    quality: int = 85,
    preserve_transparency: bool = True
) -> Tuple[Optional[bytes], Optional[int], Optional[int], bool, Optional[bytes]]:
    """Resample an image to target DPI.
    
    Returns:
        Tuple of (data, width, height, has_alpha, alpha_data)
    """
    if current_dpi is None or current_dpi <= target_dpi:
        return None, None, None, False, None
    
    scale_factor = target_dpi / current_dpi
    new_width = max(1, int(img.width * scale_factor))
    new_height = max(1, int(img.height * scale_factor))
    
    resampled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    has_alpha = False
    alpha_data = None
    
    if resampled.mode in ("RGBA", "PA") or (resampled.mode == "P" and "transparency" in resampled.info):
        if resampled.mode == "P":
            resampled = resampled.convert("RGBA")
        elif resampled.mode == "PA":
            resampled = resampled.convert("RGBA")
        
        if resampled.mode == "RGBA" and preserve_transparency:
            alpha_channel = resampled.split()[3]
            alpha_extrema = alpha_channel.getextrema()
            if alpha_extrema != (255, 255):
                has_alpha = True
                alpha_data = alpha_channel.tobytes()
                rgb_image = resampled.convert("RGB")
            else:
                rgb_image = resampled.convert("RGB")
        else:
            background = Image.new("RGB", resampled.size, (255, 255, 255))
            background.paste(resampled, mask=resampled.split()[-1])
            rgb_image = background
        resampled = rgb_image
    elif resampled.mode == "LA":
        if preserve_transparency:
            alpha_channel = resampled.split()[1]
            alpha_extrema = alpha_channel.getextrema()
            if alpha_extrema != (255, 255):
                has_alpha = True
                alpha_data = alpha_channel.tobytes()
        resampled = resampled.convert("L")
    elif resampled.mode == "CMYK":
        resampled = resampled.convert("RGB")
    elif resampled.mode == "1":
        resampled = resampled.convert("L")
    elif resampled.mode == "P":
        resampled = resampled.convert("RGB")
    
    output = io.BytesIO()
    
    if has_alpha:
        raw_data = resampled.tobytes()
        return raw_data, new_width, new_height, has_alpha, alpha_data
    else:
        if resampled.mode == "L":
            resampled.save(output, format="JPEG", quality=quality, optimize=True)
        else:
            if resampled.mode != "RGB":
                resampled = resampled.convert("RGB")
            resampled.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue(), new_width, new_height, False, None


def resample_pdf(
    input_path: str,
    output_path: str,
    dpi: int = 150,
    quality: int = 85,
    verbose: bool = False
) -> ResampleResult:
    """Resample images in a PDF to reduce file size.
    
    This is the main function for library usage. It resamples all images in the
    input PDF that are above the target DPI and saves the result to the output path.
    
    Args:
        input_path: Path to the input PDF file.
        output_path: Path where the output PDF will be saved.
        dpi: Target DPI for images (default: 150). Images at or below this DPI
            will not be resampled.
        quality: JPEG quality for resampled images, 1-100 (default: 85).
        verbose: If True, print progress information to stdout.
    
    Returns:
        ResampleResult containing statistics about the operation.
    
    Raises:
        FileProcessingError: If there is an error processing the PDF.
        InvalidParameterError: If invalid parameters are provided.
        FileNotFoundError: If the input file does not exist.
    
    Example:
        >>> from pdf_resampler import resample_pdf
        >>> result = resample_pdf("input.pdf", "output.pdf", dpi=100)
        >>> print(f"Reduced by {result.reduction_percent:.1f}%")
    """
    if dpi < 1:
        raise InvalidParameterError("DPI must be at least 1")
    if quality < 1 or quality > 100:
        raise InvalidParameterError("Quality must be between 1 and 100")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    original_size = input_path.stat().st_size
    
    if verbose:
        print(f"Opening: {input_path}")
        print(f"Target DPI: {dpi}")
        print(f"JPEG Quality: {quality}")
        print("-" * 50)
    
    try:
        pdf = pikepdf.Pdf.open(input_path, allow_overwriting_input=True)
    except Exception as e:
        raise FileProcessingError(f"Failed to open PDF: {e}")
    
    images_processed = 0
    images_skipped = 0
    image_replacements = {}
    
    if verbose:
        print("Scanning for images (including Form XObjects)...")
    
    try:
        all_images = _collect_all_images(pdf)
    except Exception as e:
        pdf.close()
        raise FileProcessingError(f"Failed to collect images: {e}")
    
    if verbose:
        print(f"Found {len(all_images)} unique image(s)")
        print("-" * 50)
    
    for objgen, (name, obj, location, render_w_pts, render_h_pts) in all_images.items():
        try:
            width_px = int(obj.get("/Width", 0))
            height_px = int(obj.get("/Height", 0))
            
            if width_px == 0 or height_px == 0:
                continue
            
            render_w_inches = render_w_pts / 72.0
            render_h_inches = render_h_pts / 72.0
            
            if render_w_inches > 0 and render_h_inches > 0:
                current_dpi_x = width_px / render_w_inches
                current_dpi_y = height_px / render_h_inches
                current_dpi = max(current_dpi_x, current_dpi_y)
            else:
                current_dpi = 300
            
            if verbose:
                print(f"Processing image '{name}' ({location}):")
                print(f"  Pixels: {width_px}x{height_px}")
                print(f"  Rendered: {render_w_pts:.1f}x{render_h_pts:.1f} pts ({render_w_inches:.2f}x{render_h_inches:.2f} in)")
                print(f"  Effective DPI: {current_dpi:.0f}")
            
            if current_dpi <= dpi:
                if verbose:
                    print(f"  Skipping (already at or below target DPI)")
                images_skipped += 1
                continue
            
            try:
                pdf_image = PdfImage(obj)
                pil_image = _extract_image_with_pdfimage(pdf_image, obj)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not create PdfImage: {e}")
                pil_image = None
            
            if pil_image is None:
                if verbose:
                    print(f"  Skipping (could not extract)")
                images_skipped += 1
                continue
            
            new_data, new_width, new_height, has_alpha, alpha_data = _resample_image(
                pil_image, dpi, current_dpi, quality
            )
            
            if new_data is None:
                if verbose:
                    print(f"  Skipping (no resampling needed)")
                images_skipped += 1
                continue
            
            if verbose:
                new_dpi_x = new_width / render_w_inches if render_w_inches > 0 else 0
                new_dpi_y = new_height / render_h_inches if render_h_inches > 0 else 0
                new_dpi = max(new_dpi_x, new_dpi_y)
                format_str = "PNG+Alpha" if has_alpha else "JPEG"
                print(f"  Resampled: {new_width}x{new_height}px @ {new_dpi:.0f} DPI ({len(new_data)/1024:.1f} KB, {format_str})")
            
            if has_alpha:
                compressed_data = zlib.compress(new_data, level=9)
                new_image = pikepdf.Stream(pdf, compressed_data)
                new_image["/Type"] = pikepdf.Name("/XObject")
                new_image["/Subtype"] = pikepdf.Name("/Image")
                new_image["/Width"] = new_width
                new_image["/Height"] = new_height
                new_image["/BitsPerComponent"] = 8
                new_image["/Filter"] = pikepdf.Name("/FlateDecode")
                
                compressed_alpha = zlib.compress(alpha_data, level=9)
                smask = pikepdf.Stream(pdf, compressed_alpha)
                smask["/Type"] = pikepdf.Name("/XObject")
                smask["/Subtype"] = pikepdf.Name("/Image")
                smask["/Width"] = new_width
                smask["/Height"] = new_height
                smask["/BitsPerComponent"] = 8
                smask["/ColorSpace"] = pikepdf.Name("/DeviceGray")
                smask["/Filter"] = pikepdf.Name("/FlateDecode")
                new_image["/SMask"] = smask
                
                if pil_image.mode == "L" or pil_image.mode == "LA":
                    new_image["/ColorSpace"] = pikepdf.Name("/DeviceGray")
                else:
                    new_image["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
            else:
                new_image = pikepdf.Stream(pdf, new_data)
                new_image["/Type"] = pikepdf.Name("/XObject")
                new_image["/Subtype"] = pikepdf.Name("/Image")
                new_image["/Width"] = new_width
                new_image["/Height"] = new_height
                new_image["/BitsPerComponent"] = 8
                new_image["/Filter"] = pikepdf.Name("/DCTDecode")
                
                if pil_image.mode == "L":
                    new_image["/ColorSpace"] = pikepdf.Name("/DeviceGray")
                else:
                    new_image["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
            
            image_replacements[objgen] = new_image
            images_processed += 1
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Error processing image '{name}': {e}")
            images_skipped += 1
    
    if verbose and image_replacements:
        print("-" * 50)
        print("Updating image references (including Form XObjects)...")
    
    _update_all_image_references(pdf, image_replacements)
    
    if verbose:
        print("-" * 50)
        print(f"Saving to: {output_path}")
    
    try:
        pdf.remove_unreferenced_resources()
        pdf.save(
            output_path, 
            compress_streams=True, 
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
            recompress_flate=True
        )
        pdf.close()
    except Exception as e:
        raise FileProcessingError(f"Failed to save PDF: {e}")
    
    new_size = output_path.stat().st_size
    
    result = ResampleResult(
        images_processed=images_processed,
        images_skipped=images_skipped,
        original_size=original_size,
        new_size=new_size
    )
    
    if verbose:
        print(f"\nResults:")
        print(f"  Images processed: {result.images_processed}")
        print(f"  Images skipped: {result.images_skipped}")
        print(f"  Original size: {result.original_size_kb:.1f} KB")
        print(f"  New size: {result.new_size_kb:.1f} KB")
        print(f"  Reduction: {result.reduction_percent:.1f}%")
    
    return result

