from upsert_helpers import add_text_document
from pathlib import Path
import fitz
from typing import Dict, Any

def add_pdf_document(pdf_path: str, metadata: Dict[str, Any], add_text_func):
    """
    Extract text from PDF and upsert using provided add_text function.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    text = text.strip()

    meta = dict(metadata)
    meta["file_path"] = str(Path(pdf_path).resolve())
    meta["file_name"] = Path(pdf_path).name

    add_text_func(text, meta)
    print(f"Inserted PDF doc '{meta['file_name']}'")
