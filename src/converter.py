"""
Converter Module
Coordinates the conversion from PDF/Images to PowerPoint.
"""

import os
from pathlib import Path
from typing import List, Optional, Callable, Union
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np

from .pdf_processor import PDFProcessor, PageContent, ExtractedText, ExtractedImage
from .image_processor import ImageProcessor, OCRResult, DetectedShape
from .color_detector import ColorDetector, ColoredObject, ColorType
from .pptx_generator import (
    PPTXGenerator, SlideContent, TextElement, ShapeElement, ImageElement, ShapeType
)
from .settings import get_settings, AppSettings


@dataclass
class ConversionProgress:
    """Progress information for conversion."""
    current_step: int
    total_steps: int
    step_name: str
    percentage: float


ProgressCallback = Callable[[ConversionProgress], None]


class Converter:
    """Main converter class for PDF/Image to PowerPoint conversion."""

    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    SUPPORTED_PDF_EXTENSION = '.pdf'

    def __init__(self, settings: Optional[AppSettings] = None):
        """
        Initialize converter.

        Args:
            settings: Application settings. If None, uses global settings.
        """
        self.settings = settings if settings else get_settings()
        self.image_processor = ImageProcessor(self.settings.tesseract_path or None)
        self.color_detector = ColorDetector()

        # Set color ranges from settings
        from .settings import get_settings_manager
        color_ranges = get_settings_manager().get_color_ranges()
        if color_ranges:
            self.color_detector.set_color_ranges(color_ranges)

    def is_pdf(self, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == self.SUPPORTED_PDF_EXTENSION

    def is_image(self, file_path: str) -> bool:
        """Check if file is a supported image."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS

    def convert(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback] = None
    ) -> bool:
        """
        Convert input file to PowerPoint.

        Args:
            input_path: Path to input file (PDF or image).
            output_path: Path to output PPTX file.
            progress_callback: Optional progress callback.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.is_pdf(input_path):
                return self._convert_pdf(input_path, output_path, progress_callback)
            elif self.is_image(input_path):
                return self._convert_image(input_path, output_path, progress_callback)
            else:
                print(f"Unsupported file type: {input_path}")
                return False
        except Exception as e:
            print(f"Conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _report_progress(
        self,
        callback: Optional[ProgressCallback],
        current: int,
        total: int,
        step_name: str
    ):
        """Report progress to callback."""
        if callback:
            callback(ConversionProgress(
                current_step=current,
                total_steps=total,
                step_name=step_name,
                percentage=(current / total) * 100 if total > 0 else 0
            ))

    def _convert_pdf(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback]
    ) -> bool:
        """Convert PDF to PowerPoint."""
        processor = PDFProcessor(input_path)

        if not processor.open():
            return False

        try:
            generator = PPTXGenerator()
            page_count = processor.get_page_count()
            total_steps = page_count * 3  # Extract, process, generate per page

            for page_num in range(page_count):
                step_base = page_num * 3

                # Step 1: Extract content
                self._report_progress(
                    progress_callback,
                    step_base + 1, total_steps,
                    f"Extracting page {page_num + 1}/{page_count}"
                )

                page_content = processor.extract_page_content(page_num)
                if not page_content:
                    continue

                # Render page as image for color detection
                page_image = processor.render_page_as_image(page_num, self.settings.processing.dpi)

                # Step 2: Process content - detect colored objects
                self._report_progress(
                    progress_callback,
                    step_base + 2, total_steps,
                    f"Processing page {page_num + 1}/{page_count}"
                )

                colored_objects = []
                if self.settings.processing.detect_colors and page_image:
                    cv_image = self.image_processor.pil_to_cv2(page_image)
                    colored_objects = self.color_detector.detect_colors(
                        cv_image,
                        min_area=self.settings.processing.min_color_area
                    )

                # Step 3: Generate slide
                self._report_progress(
                    progress_callback,
                    step_base + 3, total_steps,
                    f"Generating slide {page_num + 1}/{page_count}"
                )

                self._create_slide_from_page(
                    generator,
                    page_content,
                    page_image,
                    colored_objects
                )

            # Save presentation
            generator.save(output_path)
            return True

        finally:
            processor.close()

    def _convert_image(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[ProgressCallback]
    ) -> bool:
        """Convert image to PowerPoint."""
        total_steps = 4

        # Step 1: Load image
        self._report_progress(progress_callback, 1, total_steps, "Loading image")

        image = self.image_processor.load_image(input_path)
        if image is None:
            return False

        pil_image = self.image_processor.cv2_to_pil(image)
        height, width = image.shape[:2]

        # Step 2: Extract text with OCR
        self._report_progress(progress_callback, 2, total_steps, "Extracting text (OCR)")

        ocr_results = []
        if self.settings.processing.ocr_language:
            ocr_results = self.image_processor.extract_text_ocr(
                image,
                lang=self.settings.processing.ocr_language
            )

        # Step 3: Detect colors and shapes
        self._report_progress(progress_callback, 3, total_steps, "Detecting objects")

        colored_objects = []
        if self.settings.processing.detect_colors:
            colored_objects = self.color_detector.detect_colors(
                image,
                min_area=self.settings.processing.min_color_area
            )

        shapes = []
        if self.settings.processing.detect_shapes:
            shapes = self.image_processor.detect_shapes(image)

        # Step 4: Generate PowerPoint
        self._report_progress(progress_callback, 4, total_steps, "Generating PowerPoint")

        generator = PPTXGenerator()
        self._create_slide_from_image(
            generator,
            pil_image,
            width, height,
            ocr_results,
            colored_objects,
            shapes
        )

        generator.save(output_path)
        return True

    def _create_slide_from_page(
        self,
        generator: PPTXGenerator,
        page_content: PageContent,
        page_image: Optional[Image.Image],
        colored_objects: List[ColoredObject]
    ):
        """Create a slide from PDF page content with actual PowerPoint objects."""
        elements = []
        source_width = page_content.width
        source_height = page_content.height

        # Calculate image scale factor if page was rendered as image
        img_scale_x = 1.0
        img_scale_y = 1.0
        if page_image:
            img_width, img_height = page_image.size
            img_scale_x = source_width / img_width
            img_scale_y = source_height / img_height

        # Group texts by proximity to form text blocks
        text_blocks = self._group_texts_into_blocks(page_content.texts)

        # Add text elements from PDF
        for block in text_blocks:
            # Calculate bounding box for the block
            min_x = min(t.bbox[0] for t in block)
            min_y = min(t.bbox[1] for t in block)
            max_x = max(t.bbox[2] for t in block)
            max_y = max(t.bbox[3] for t in block)

            # Combine text from all items in block
            block_text = " ".join(t.text for t in block)

            # Get average font size
            avg_font_size = sum(t.font_size for t in block) / len(block)

            x_in, y_in = generator.convert_position_to_inches(
                min_x, min_y, source_width, source_height
            )
            w_in, h_in = generator.convert_size_to_inches(
                max_x - min_x, max_y - min_y, source_width, source_height
            )

            # Ensure minimum size
            w_in = max(w_in, 0.5)
            h_in = max(h_in, 0.3)

            elements.append(TextElement(
                x=x_in, y=y_in,
                width=w_in, height=h_in,
                text=block_text,
                font_size=min(avg_font_size, 24),  # Cap font size
                font_name=self.settings.font.default_font,
                font_color=(0, 0, 0)
            ))

        # Add images from PDF
        for img in page_content.images:
            x0, y0, x1, y1 = img.bbox
            x_in, y_in = generator.convert_position_to_inches(
                x0, y0, source_width, source_height
            )
            w_in, h_in = generator.convert_size_to_inches(
                x1 - x0, y1 - y0, source_width, source_height
            )

            elements.append(ImageElement(
                x=x_in, y=y_in,
                width=max(w_in, 0.5),
                height=max(h_in, 0.5),
                image=img.image
            ))

        # Add colored objects as shapes with fill
        for obj in colored_objects:
            x, y, w, h = obj.bbox
            # Convert from image coordinates to PDF coordinates
            x *= img_scale_x
            y *= img_scale_y
            w *= img_scale_x
            h *= img_scale_y

            x_in, y_in = generator.convert_position_to_inches(
                x, y, source_width, source_height
            )
            w_in, h_in = generator.convert_size_to_inches(
                w, h, source_width, source_height
            )

            # Determine shape type based on detected shape
            shape_type = self._detect_shape_type_from_contour(obj.contour)

            elements.append(ShapeElement(
                x=x_in, y=y_in,
                width=max(w_in, 0.2),
                height=max(h_in, 0.2),
                shape_type=shape_type,
                fill_color=obj.rgb_color,  # Fill with detected color
                line_color=self._darken_color(obj.rgb_color),  # Darker border
                line_width=1
            ))

        content = SlideContent(elements=elements)
        generator.add_slide_with_content(content)

    def _create_slide_from_image(
        self,
        generator: PPTXGenerator,
        image: Image.Image,
        width: int,
        height: int,
        ocr_results: List[OCRResult],
        colored_objects: List[ColoredObject],
        shapes: List[DetectedShape]
    ):
        """Create a slide from image content with actual PowerPoint objects."""
        elements = []

        # Group OCR results into text blocks
        text_blocks = self._group_ocr_into_blocks(ocr_results)

        # Add text elements from OCR
        for block in text_blocks:
            # Calculate bounding box for the block
            min_x = min(r.bbox[0] for r in block)
            min_y = min(r.bbox[1] for r in block)
            max_x = max(r.bbox[0] + r.bbox[2] for r in block)
            max_y = max(r.bbox[1] + r.bbox[3] for r in block)

            # Combine text
            block_text = " ".join(r.text for r in block)

            x_in, y_in = generator.convert_position_to_inches(
                min_x, min_y, width, height
            )
            w_in, h_in = generator.convert_size_to_inches(
                max_x - min_x, max_y - min_y, width, height
            )

            # Ensure minimum size
            w_in = max(w_in, 0.5)
            h_in = max(h_in, 0.3)

            elements.append(TextElement(
                x=x_in, y=y_in,
                width=w_in, height=h_in,
                text=block_text,
                font_size=self.settings.font.body_size,
                font_name=self.settings.font.default_font,
                font_color=(0, 0, 0)
            ))

        # Add detected shapes as PowerPoint shapes
        for shape in shapes:
            x, y, w, h = shape.bbox
            x_in, y_in = generator.convert_position_to_inches(x, y, width, height)
            w_in, h_in = generator.convert_size_to_inches(w, h, width, height)

            shape_type = self._convert_detected_shape_type(shape.shape_type)

            elements.append(ShapeElement(
                x=x_in, y=y_in,
                width=max(w_in, 0.2),
                height=max(h_in, 0.2),
                shape_type=shape_type,
                fill_color=None,  # No fill for detected shapes
                line_color=(0, 0, 0),
                line_width=1
            ))

        # Add colored objects as shapes with fill
        for obj in colored_objects:
            x, y, w, h = obj.bbox
            x_in, y_in = generator.convert_position_to_inches(x, y, width, height)
            w_in, h_in = generator.convert_size_to_inches(w, h, width, height)

            shape_type = self._detect_shape_type_from_contour(obj.contour)

            elements.append(ShapeElement(
                x=x_in, y=y_in,
                width=max(w_in, 0.2),
                height=max(h_in, 0.2),
                shape_type=shape_type,
                fill_color=obj.rgb_color,
                line_color=self._darken_color(obj.rgb_color),
                line_width=1
            ))

        content = SlideContent(elements=elements)
        generator.add_slide_with_content(content)

    def _group_texts_into_blocks(
        self,
        texts: List[ExtractedText],
        line_threshold: float = 5.0
    ) -> List[List[ExtractedText]]:
        """Group text elements into blocks based on proximity."""
        if not texts:
            return []

        # Sort by y position then x position
        sorted_texts = sorted(texts, key=lambda t: (t.bbox[1], t.bbox[0]))

        blocks = []
        current_block = [sorted_texts[0]]
        current_y = sorted_texts[0].bbox[1]

        for text in sorted_texts[1:]:
            # Check if this text is on the same line (within threshold)
            if abs(text.bbox[1] - current_y) <= line_threshold:
                current_block.append(text)
            else:
                # Start a new block
                blocks.append(current_block)
                current_block = [text]
                current_y = text.bbox[1]

        # Add the last block
        if current_block:
            blocks.append(current_block)

        return blocks

    def _group_ocr_into_blocks(
        self,
        ocr_results: List[OCRResult],
        line_threshold: int = 10
    ) -> List[List[OCRResult]]:
        """Group OCR results into text blocks based on proximity."""
        if not ocr_results:
            return []

        # Filter out low confidence results
        filtered = [r for r in ocr_results if r.confidence > 0.5]
        if not filtered:
            return []

        # Sort by y position then x position
        sorted_results = sorted(filtered, key=lambda r: (r.bbox[1], r.bbox[0]))

        blocks = []
        current_block = [sorted_results[0]]
        current_y = sorted_results[0].bbox[1]

        for result in sorted_results[1:]:
            # Check if this result is on the same line (within threshold)
            if abs(result.bbox[1] - current_y) <= line_threshold:
                current_block.append(result)
            else:
                # Start a new block
                blocks.append(current_block)
                current_block = [result]
                current_y = result.bbox[1]

        # Add the last block
        if current_block:
            blocks.append(current_block)

        return blocks

    def _detect_shape_type_from_contour(self, contour: np.ndarray) -> ShapeType:
        """Detect shape type from contour."""
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        if vertices == 3:
            return ShapeType.TRIANGLE
        elif vertices == 4:
            # Check aspect ratio for square vs rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 1
            if 0.9 <= aspect_ratio <= 1.1:
                return ShapeType.RECTANGLE  # Square-ish
            return ShapeType.RECTANGLE
        elif vertices == 5:
            return ShapeType.PENTAGON
        elif vertices == 6:
            return ShapeType.HEXAGON
        else:
            # Check circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:
                    return ShapeType.OVAL
            return ShapeType.RECTANGLE

    def _convert_detected_shape_type(self, shape_type_str: str) -> ShapeType:
        """Convert detected shape type string to ShapeType enum."""
        mapping = {
            'rectangle': ShapeType.RECTANGLE,
            'square': ShapeType.RECTANGLE,
            'circle': ShapeType.OVAL,
            'triangle': ShapeType.TRIANGLE,
            'pentagon': ShapeType.PENTAGON,
            'hexagon': ShapeType.HEXAGON,
            'polygon': ShapeType.RECTANGLE,
        }
        return mapping.get(shape_type_str, ShapeType.RECTANGLE)

    def _darken_color(self, rgb: tuple, factor: float = 0.7) -> tuple:
        """Darken a color for use as border."""
        return (
            int(rgb[0] * factor),
            int(rgb[1] * factor),
            int(rgb[2] * factor)
        )

    def _get_shape_type_for_color(self, color_type: ColorType) -> ShapeType:
        """Get appropriate shape type for a color type."""
        # Map semantic color types to shapes
        shape_map = {
            ColorType.RED: ShapeType.DIAMOND,      # Warning
            ColorType.YELLOW: ShapeType.TRIANGLE,  # Caution
            ColorType.GREEN: ShapeType.OVAL,       # Success
            ColorType.BLUE: ShapeType.RECTANGLE,   # Info
            ColorType.ORANGE: ShapeType.PENTAGON,  # Warning (mild)
            ColorType.PURPLE: ShapeType.STAR,      # Special
        }
        return shape_map.get(color_type, ShapeType.RECTANGLE)


def convert_file(
    input_path: str,
    output_path: str,
    progress_callback: Optional[ProgressCallback] = None
) -> bool:
    """
    Convenience function to convert a file.

    Args:
        input_path: Path to input file.
        output_path: Path to output PPTX file.
        progress_callback: Optional progress callback.

    Returns:
        True if successful.
    """
    converter = Converter()
    return converter.convert(input_path, output_path, progress_callback)
