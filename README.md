# 🚀 Modular OCR Agent Demo

A powerful, agentic OCR system built with **Google ADK** and **Gemini 2.0 Flash**. This project features tiered agents that autonomously use vision tools to handle everything from simple images to complex, multi-page research papers.

## ✨ Key Features
- **Multi-Agent Support**: Choose between a standard agent and a customized agent with advanced vision tools.
- **Advanced Tools**: Preprocessing (contrast/noise), Tiling (high-res cropping), and PDF Page Splitting.
- **Multi-Mode Operation**: Switch between `OCR` (full text) and `Description` (visual summary) modes.
- **Incremental Versioning**: Never lose your history—reports are saved with automatic `_v1`, `_v2` suffixes.
- **Targeted Processing**: Process a single file using the `--file` flag.
- **PDF Ready**: Supports multi-page PDF documents.

---

## 🛠️ Getting Started

### 1. Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) installed.

### 2. Environment Setup
Create a `.env` file in the root directory and add your Google API key:
```env
GOOGLE_API_KEY=YOUR_API_KEY_HERE
```

### 3. Folder Structure
- **`input/`**: Drop your images (`.png`, `.jpg`, `.pdf`) here.
- **`output/`**: Your markdown reports will appear here.

---

## 🚀 Usage Guide

### Basic: Process everything in `input/`
```bash
uv run run_agent.py
```

### Advanced: Choose your Agent
- **Standard (`ocr_agent`)**: Good for most documents.
- **Customized (`ocr_customed_agent`)**: Uses advanced sharpening and tiling for complex files.
```bash
uv run run_agent.py --agent ocr_customed_agent
```

### Targeted: Process a specific file
```bash
uv run run_agent.py --file "my_document.pdf"
```

### Vision: Get a summary instead of extraction
```bash
uv run run_agent.py description --file "poster.png"
```

---

## 🧩 Agent Capabilities

### 🖼️ Advanced Vision Tools (Customized Agent Only)
- **Preprocessing**: Fixes low-contrast and noisy images before extraction.
- **Tiling**: Splits high-resolution images into smaller "chunks" to maintain detail.
- **Zooming**: Crops into specific regions for fine-grained OCR.

### 📄 Multi-Page PDF Handling
The agent automatically detects the page count of PDFs. For large documents (>10 pages), the customized agent will use its **Chunking Tool** to process the paper in 5-10 page "bites" to stay under the model's token limit.

---

## 📝 Troubleshooting & Tips
- **Rate Limits**: If you see a `429 RESOURCE_EXHAUSTED` error, wait 60 seconds and try again on a single file using the `--file` flag.
- **Token Limits**: For massive documents (1.9M+ tokens), use the **`ocr_customed_agent`** to ensure the document is chunked correctly.
