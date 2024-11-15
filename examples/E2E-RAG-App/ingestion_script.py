import json
import logging
from pathlib import Path

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

def get_document_files(input_dir: Path) -> list[Path]:
    """
    Recursively scan directory for document files.
    Returns a list of Path objects for supported document types.
    """
    supported_extensions = {'.pdf', '.docx', '.pptx'}
    document_files = []
    
    # Recursively walk through all directories
    for path in input_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            document_files.append(path)
    
    return document_files

def main():
    # Define input and output directories relative to current directory
    input_dir = Path("DATA")
    output_dir = Path("OUTPUT")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all document files recursively
    input_paths = get_document_files(input_dir)
    
    if not input_paths:
        print("No documents found in DATA directory!")
        return
        
    print(f"Found {len(input_paths)} documents to process:")
    for path in input_paths:
        print(f"- {path}")
    
    # Configure document converter
    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline, 
                backend=PyPdfiumDocumentBackend
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline
            ),
        },
    )
    
    # Process all documents
    conv_results = doc_converter.convert_all(input_paths)
    
    # Save results
    for res in conv_results:
        # Preserve directory structure in output
        relative_path = res.input.file.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create output filenames
        md_path = output_subdir / f"{res.input.file.stem}.md"
        json_path = output_subdir / f"{res.input.file.stem}.json"
        
        print(
            f"Converting: {res.input.file}"
            f"\nSaving to: {md_path}"
        )
        
        # Save markdown version
        with md_path.open("w", encoding='utf-8') as fp:
            fp.write(res.document.export_to_markdown())
            
        # Save JSON version
        with json_path.open("w", encoding='utf-8') as fp:
            json.dump(res.document.export_to_dict(), fp, indent=2)

if __name__ == "__main__":
    main()
