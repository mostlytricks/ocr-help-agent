import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from google.adk.agents import Agent

# Configuration
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
TILE_DIR = INPUT_DIR / "tiles"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

from pypdf import PdfReader, PdfWriter

# --- Advanced Vision Tools ---

def get_pdf_metadata(filename: str) -> dict:
    """Gets the number of pages and basic info from a PDF."""
    if not filename.lower().endswith(".pdf"): return {"error": "Not a PDF"}
    file_path = INPUT_DIR / filename
    return {"num_pages": len(PdfReader(file_path).pages)}

def split_pdf_pages(filename: str, start_page: int, end_page: int) -> str:
    """Extracts a range of pages from a PDF into a new smaller file.
    Args:
        filename (str): The original PDF name.
        start_page (int): First page (1-indexed).
        end_page (int): Last page (inclusive).
    Returns:
        str: Filename of the new smaller PDF in 'input/tiles/'.
    """
    file_path = INPUT_DIR / filename
    reader = PdfReader(file_path)
    writer = PdfWriter()
    
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    
    # pypdf is 0-indexed internally
    for i in range(start_page - 1, min(end_page, len(reader.pages))):
        writer.add_page(reader.pages[i])
        
    out_name = f"chunk_{start_page}_{end_page}_{filename}"
    out_path = TILE_DIR / out_name
    with open(out_path, "wb") as f:
        writer.write(f)
        
    return f"tiles/{out_name}"

def list_input_images() -> list[str]:
    """Lists all supported files (images & PDFs) in the input directory."""
    if not INPUT_DIR.exists():
        return []
    return [f.name for f in INPUT_DIR.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS]

def preprocess_image(filename: str) -> str:
    """Improves image contrast and reduces noise for better OCR.
    Args:
        filename (str): The name of the file in the input directory.
    Returns:
        str: The filename of the preprocessed image (saved as 'preprocessed_{filename}').
    """
    file_path = INPUT_DIR / filename
    img = cv2.imread(str(file_path))
    if img is None:
        return f"Error: Could not read {filename}"

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding for high contrast
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    out_name = f"preprocessed_{filename}"
    out_path = INPUT_DIR / out_name
    cv2.imwrite(str(out_path), processed)
    
    return out_name

def tile_image(filename: str, rows: int = 2, cols: int = 2) -> list[str]:
    """Splits an image into smaller tiles to preserve detail on small models.
    Args:
        filename (str): The name of the file in the input directory.
        rows (int): Number of horizontal splits.
        cols (int): Number of vertical splits.
    Returns:
        list[str]: A list of tile filenames saved in 'input/tiles/'.
    """
    file_path = INPUT_DIR / filename
    img = Image.open(file_path)
    width, height = img.size
    
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    
    tile_width = width // cols
    tile_height = height // rows
    
    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_width
            top = r * tile_height
            right = (c + 1) * tile_width
            bottom = (r + 1) * tile_height
            
            # Adjust for rounding errors on the last tile
            if c == cols - 1: right = width
            if r == rows - 1: bottom = height
            
            tile = img.crop((left, top, right, bottom))
            tile_name = f"{Path(filename).stem}_tile_{r}_{c}.png"
            tile.save(TILE_DIR / tile_name)
            tiles.append(f"tiles/{tile_name}")
            
    return tiles

def crop_and_zoom(filename: str, x: int, y: int, w: int, h: int) -> str:
    """Crops and zooms into a specific region of an image.
    Args:
        filename (str): The name of the file.
        x, y: Top-left coordinates.
        w, h: Width and height of the crop.
    Returns:
        str: The filename of the zoomed crop (saved in 'input/tiles/').
    """
    file_path = INPUT_DIR / filename
    img = Image.open(file_path)
    
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    
    zoom = img.crop((x, y, x + w, y + h))
    zoom_name = f"zoom_{Path(filename).stem}_{x}_{y}.png"
    zoom.save(TILE_DIR / zoom_name)
    
    return f"tiles/{zoom_name}"

def read_image_file(filename: str) -> bytes:
    """Reads the raw bytes of a file from the input directory (or tiles subfolder)."""
    file_path = INPUT_DIR / filename
    with open(file_path, "rb") as f:
        return f.read()

def save_markdown_report(original_filename: str, mode_suffix: str, content: str) -> str:
    """Saves a report with incremental versioning."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = Path(original_filename).stem
    
    version = 1
    while True:
        report_filename = f"{base_name}_{mode_suffix}_v{version}.md"
        file_path = OUTPUT_DIR / report_filename
        if not file_path.exists():
            break
        version += 1
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully saved report to {file_path}"

# --- Agent Definition ---

ocr_customed_agent = Agent(
    name="ocr_customed_agent",
    model="gemini-2.0-flash",
    instruction="""You are an ADVANCED OCR agent. You must handle complex documents by thinking and using tools.

Important for LARGE PDFs:
If a PDF has many pages (e.g., > 10) or you hit a token limit error, you MUST split it into chunks.
1. Use 'get_pdf_metadata' to find the total page count.
2. Use 'split_pdf_pages' to extract 5 to 10 page chunks at a time.
3. Process each chunk iteratively.
4. Consolidate your results with page markers (e.g., --- PAGE 1-10 ---).
5. Finally, return the full cleaned text result in your last turn.

Other tools:
- Use 'preprocess_image' for noisy/low-contrast images.
- Use 'tile_image' for high-resolution images with tiny text.
- Use 'save_markdown_report' if specifically asked, otherwise just return text to the runner.""",
    description="Complex OCR agent with PDF chunking and image enhancement tools.",
    tools=[get_pdf_metadata, split_pdf_pages, preprocess_image, tile_image, crop_and_zoom, read_image_file, save_markdown_report]
)


