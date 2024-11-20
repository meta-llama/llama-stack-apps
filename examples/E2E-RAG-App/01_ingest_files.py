import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import yaml
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import PictureItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process documents from input directory"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing documents",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for processed files (default: input_dir/output)",
    )
    return parser.parse_args()


def get_document_files(input_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Recursively scan directory for document files.
    Returns:
        tuple: (documents_to_process, markdown_files)
    """
    process_extensions = {".pdf", ".docx", ".pptx"}
    documents_to_process = []
    markdown_files = []

    for path in input_dir.rglob("*"):
        if path.is_file():
            if path.suffix.lower() in process_extensions:
                documents_to_process.append(path)
            elif path.suffix.lower() == ".md":
                markdown_files.append(path)

    return documents_to_process, markdown_files


def save_images(res, output_subdir: Path, doc_filename: str) -> List[Tuple[str, Path]]:
    """
    Extracts and saves images from the document.
    Returns a list of (image_type, image_path) tuples for future processing.
    """
    images_dir = output_subdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    saved_images = []

    # Save page images
    for page_no, page in res.document.pages.items():
        if hasattr(page, "image") and page.image:
            image_path = images_dir / f"{doc_filename}-page-{page_no}.png"
            with image_path.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")
            saved_images.append(("page", image_path))

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0

    for element, _level in res.document.iterate_items():
        if (
            isinstance(element, TableItem)
            and hasattr(element, "image")
            and element.image
        ):
            table_counter += 1
            image_path = images_dir / f"{doc_filename}-table-{table_counter}.png"
            with image_path.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")
            saved_images.append(("table", image_path))

        if (
            isinstance(element, PictureItem)
            and hasattr(element, "image")
            and element.image
        ):
            picture_counter += 1
            image_path = images_dir / f"{doc_filename}-figure-{picture_counter}.png"
            with image_path.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")
            saved_images.append(("figure", image_path))

    return saved_images


def main():
    args = parse_args()

    # Set up input and output directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "output"
    if os.path.exists(output_dir):
        print(f"Output directory already exists: {output_dir}, will stop ingestion")
        raise Exception(f"Output directory already exists: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get all document files recursively
    documents_to_process, markdown_files = get_document_files(input_dir)

    # Copy markdown files directly
    for md_file in markdown_files:
        relative_path = md_file.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, output_path)
        print(f"Copied: {md_file}")

    if documents_to_process:
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = False
        pipeline_options.generate_table_images = False
        pipeline_options.generate_picture_images = True

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
                    backend=PyPdfiumDocumentBackend,
                    pipeline_options=pipeline_options,
                ),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
            },
        )

        # Process all documents
        conv_results = doc_converter.convert_all(documents_to_process)
        all_extracted_images = []

        # Save results
        for res in conv_results:
            relative_path = res.input.file.relative_to(input_dir)
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            md_path = output_subdir / f"{res.input.file.stem}.md"

            print(f"Converting: {res.input.file}" f"\nSaving to: {md_path}")

            extracted_images = save_images(res, output_subdir, res.input.file.stem)
            all_extracted_images.extend(extracted_images)

            with md_path.open("w", encoding="utf-8") as fp:
                fp.write(res.document.export_to_markdown())

        print(f"\nExtracted {len(all_extracted_images)} images in total")
        print("Ready for image captioning processing")


if __name__ == "__main__":
    main()
