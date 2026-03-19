import os
from pathlib import Path
from google.adk.agents import Agent

# Configuration
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
# Expanded to support PDFs
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}

from pypdf import PdfReader, PdfWriter

# --- Tools ---

def get_pdf_metadata(filename: str) -> dict:
    """Gets the number of pages from a PDF."""
    if not filename.lower().endswith(".pdf"): return {"error": "Not a PDF"}
    file_path = Path("input") / filename
    return {"num_pages": len(PdfReader(file_path).pages)}

def split_pdf_pages(filename: str, start_page: int, end_page: int) -> str:
    """Extracts a range of pages from a PDF into a new smaller file."""
    file_path = Path("input") / filename
    reader = PdfReader(file_path)
    writer = PdfWriter()
    
    # Ensure tiles dir exists
    out_dir = Path("input/tiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(start_page - 1, min(end_page, len(reader.pages))):
        writer.add_page(reader.pages[i])
        
    out_name = f"chunk_{start_page}_{end_page}_{filename}"
    out_path = out_dir / out_name
    with open(out_path, "wb") as f:
        writer.write(f)
        
    return f"tiles/{out_name}"

def list_input_images() -> list[str]:
    """Lists all supported files (images & PDFs) in the input directory."""
    if not INPUT_DIR.exists():
        return []
    return [f.name for f in INPUT_DIR.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS]

def read_image_file(filename: str) -> bytes:
    """Reads the raw bytes of a file from the input directory."""
    file_path = INPUT_DIR / filename
    with open(file_path, "rb") as f:
        return f.read()

def save_markdown_report(original_filename: str, mode_suffix: str, content: str) -> str:
    """Saves a report with incremental versioning (e.g., _v1, _v2) to prevent overwriting.
    Args:
        original_filename (str): The name of the input file.
        mode_suffix (str): The report type suffix.
        content (str): The markdown report content.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = Path(original_filename).stem
    
    # Versioning logic
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

ocr_agent = Agent(
    name="ocr_agent",
    model="gemini-2.0-flash",
    instruction="""You are a versatile OCR and Vision agent. 

**Working with LARGE PDFs (>10 pages):**
You must avoid token limit errors by processing the document in chunks.
1. Use 'get_pdf_metadata' to find the page count.
2. Use 'split_pdf_pages' to extract 5-10 pages at a time.
3. Consolidate your results and return the final text in your last turn.

**Standard OCR/Description Mode:**
1. Extract all text with high accuracy or provide a concise description.
2. Return the final CLEAN text to the runner. The runner handles file saving.

Available Tools:
- Use 'list_input_images' to see files.
- Use 'read_image_file' to get bytes.
- Use 'split_pdf_pages' and 'get_pdf_metadata' for PDFs.""",
    description="Multilingual OCR agent with PDF chunking capabilities.",
    tools=[list_input_images, read_image_file, save_markdown_report, get_pdf_metadata, split_pdf_pages]
)




