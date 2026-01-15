"""
PDF Processing Module
Extracts text and images from PDF files using PyMuPDF.
"""

import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExtractedImage:
    """Represents an extracted image from PDF."""
    image: Image.Image
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    xref: int


@dataclass
class ExtractedText:
    """Represents extracted text from PDF."""
    text: str
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_name: str


@dataclass
class TextBlock:
    """Represents a text block (paragraph-level) from PDF."""
    text: str
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float  # Average font size in block


@dataclass
class PageContent:
    """Represents all content from a single PDF page."""
    page_num: int
    width: float
    height: float
    texts: List[ExtractedText]
    text_blocks: List[TextBlock]  # Block-level text
    images: List[ExtractedImage]


class PDFProcessor:
    """Processes PDF files to extract text and images."""

    def __init__(self, pdf_path: str):
        """
        Initialize PDF processor.

        Args:
            pdf_path: Path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.doc: Optional[fitz.Document] = None

    def open(self) -> bool:
        """
        Open the PDF document.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.doc = fitz.open(self.pdf_path)
            return True
        except Exception as e:
            print(f"Error opening PDF: {e}")
            return False

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
            self.doc = None

    def get_page_count(self) -> int:
        """Get the number of pages in the PDF."""
        if self.doc:
            return len(self.doc)
        return 0

    def extract_page_content(self, page_num: int) -> Optional[PageContent]:
        """
        Extract all content from a specific page.

        Args:
            page_num: Page number (0-indexed).

        Returns:
            PageContent object containing texts and images.
        """
        if not self.doc or page_num >= len(self.doc):
            return None

        page = self.doc[page_num]
        width = page.rect.width
        height = page.rect.height

        texts = self._extract_texts(page, page_num)
        text_blocks = self._extract_text_blocks(page, page_num)
        images = self._extract_images(page, page_num)

        return PageContent(
            page_num=page_num,
            width=width,
            height=height,
            texts=texts,
            text_blocks=text_blocks,
            images=images
        )

    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """
        Extract text at block level (paragraph-like grouping).

        Args:
            page: PyMuPDF page object.
            page_num: Page number.

        Returns:
            List of TextBlock objects.
        """
        blocks = []

        # Get text blocks - returns (x0, y0, x1, y1, "text", block_no, block_type)
        raw_blocks = page.get_text("blocks")

        for block in raw_blocks:
            if len(block) >= 7 and block[6] == 0:  # Text block (type 0)
                x0, y0, x1, y1, text, block_no, block_type = block[:7]
                text = text.strip()
                if text:
                    # Estimate font size from block height and line count
                    line_count = text.count('\n') + 1
                    avg_line_height = (y1 - y0) / line_count
                    estimated_font_size = avg_line_height * 0.7  # Approximate

                    blocks.append(TextBlock(
                        text=text,
                        page_num=page_num,
                        bbox=(x0, y0, x1, y1),
                        font_size=max(8, min(estimated_font_size, 48))
                    ))

        return blocks

    def _extract_texts(self, page: fitz.Page, page_num: int) -> List[ExtractedText]:
        """
        Extract text spans from a page (fine-grained).

        Args:
            page: PyMuPDF page object.
            page_num: Page number.

        Returns:
            List of ExtractedText objects.
        """
        texts = []

        # Get text blocks with detailed info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            bbox = span.get("bbox", (0, 0, 0, 0))
                            texts.append(ExtractedText(
                                text=text,
                                page_num=page_num,
                                bbox=tuple(bbox),
                                font_size=span.get("size", 12),
                                font_name=span.get("font", "")
                            ))

        return texts

    def _extract_images(self, page: fitz.Page, page_num: int) -> List[ExtractedImage]:
        """
        Extract images from a page.

        Args:
            page: PyMuPDF page object.
            page_num: Page number.

        Returns:
            List of ExtractedImage objects.
        """
        images = []
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Get image rectangle on the page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    bbox = img_rects[0]
                    images.append(ExtractedImage(
                        image=pil_image,
                        page_num=page_num,
                        bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        xref=xref
                    ))
            except Exception as e:
                print(f"Error extracting image (xref={xref}): {e}")

        return images

    def extract_drawings(self, page_num: int) -> List[Dict]:
        """
        Extract vector drawings/shapes from a page.

        Args:
            page_num: Page number (0-indexed).

        Returns:
            List of drawing dictionaries with type and coordinates.
        """
        if not self.doc or page_num >= len(self.doc):
            return []

        page = self.doc[page_num]
        drawings = []

        try:
            paths = page.get_drawings()
            for path in paths:
                drawing = {
                    'rect': path.get('rect'),
                    'fill': path.get('fill'),
                    'color': path.get('color'),
                    'width': path.get('width', 1),
                    'items': path.get('items', [])
                }

                # Determine shape type from items
                items = path.get('items', [])
                if items:
                    first_item = items[0]
                    if first_item[0] == 're':  # Rectangle
                        drawing['type'] = 'rectangle'
                    elif first_item[0] == 'c':  # Curve
                        drawing['type'] = 'curve'
                    elif first_item[0] == 'l':  # Line
                        drawing['type'] = 'line'
                    else:
                        drawing['type'] = 'path'

                drawings.append(drawing)
        except Exception as e:
            print(f"Error extracting drawings: {e}")

        return drawings

    def extract_all_pages(self, progress_callback=None) -> List[PageContent]:
        """
        Extract content from all pages.

        Args:
            progress_callback: Optional callback function(current, total).

        Returns:
            List of PageContent objects.
        """
        pages = []
        total = self.get_page_count()

        for i in range(total):
            content = self.extract_page_content(i)
            if content:
                pages.append(content)

            if progress_callback:
                progress_callback(i + 1, total)

        return pages

    def render_page_as_image(self, page_num: int, dpi: int = 150) -> Optional[Image.Image]:
        """
        Render a page as an image.

        Args:
            page_num: Page number (0-indexed).
            dpi: Resolution in DPI.

        Returns:
            PIL Image object.
        """
        if not self.doc or page_num >= len(self.doc):
            return None

        page = self.doc[page_num]
        zoom = dpi / 72  # 72 is the default PDF resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img


def extract_pdf_content(pdf_path: str, progress_callback=None) -> List[PageContent]:
    """
    Convenience function to extract all content from a PDF.

    Args:
        pdf_path: Path to PDF file.
        progress_callback: Optional callback function(current, total).

    Returns:
        List of PageContent objects.
    """
    processor = PDFProcessor(pdf_path)
    try:
        if processor.open():
            return processor.extract_all_pages(progress_callback)
        return []
    finally:
        processor.close()
