import json
import logging
from pathlib import Path
from typing import Tuple, List

import yaml
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import TableItem, PictureItem



def get_document_files(input_dir: Path) -> list[Path]:
    """
    Recursively scan directory for document files.
    Returns a list of Path objects for supported document types.
    """
    supported_extensions = {".pdf", ".docx", ".pptx"}
    document_files = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            document_files.append(path)

    return document_files