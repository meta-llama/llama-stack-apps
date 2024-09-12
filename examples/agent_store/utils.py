import os
from typing import Dict

from pypdf import PdfReader


def load_pdfs_from_directory(directory_path) -> Dict[str, str]:
    """
    Load all PDF files from a directory and return a dictionary
    mapping file names to their contents.
    """
    pdf_files = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                pdf = PdfReader(file_path)
                pdf_files[filename] = pdf
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

    content = {}
    for name, reader in pdf_files.items():
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        content[name] = text

    return content
