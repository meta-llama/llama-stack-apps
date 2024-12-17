# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import base64
import mimetypes
from pathlib import Path

from llama_stack_client import LlamaStackClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process document images with LlamaStack Vision API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="LlamaStack server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="LlamaStack server port (default: 5000)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing markdown files and images",
    )
    return parser.parse_args()


def encode_image_to_data_url(file_path: str) -> str:
    """Encode an image file to a data URL."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type of the file")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"


class DocumentProcessor:
    def __init__(self, host: str, port: int):
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.processed_images = {}

    async def get_image_caption(self, image_path: str) -> str:
        """Get caption for an image using LlamaStack Vision API."""
        if image_path in self.processed_images:
            return self.processed_images[image_path]

        try:
            data_url = encode_image_to_data_url(image_path)

            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": {
                            "uri": data_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": "This image comes from a scan inside a document, please provide a high level caption of what you see inside the image.",
                    },
                ],
            }

            response = await self.client.inference.chat_completion(
                messages=[message], model="Llama3.2-11B-Vision-Instruct", stream=False
            )

            caption = response.choices[0].message.content
            self.processed_images[image_path] = caption
            return caption

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    async def process_markdown_file(self, output_dir: str, md_filename: str) -> None:
        """Process a single markdown file and replace image placeholders with captions."""
        print(f"Processing: {md_filename}")

        md_path = Path(output_dir) / md_filename
        images_dir = Path(output_dir) / "images"

        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Failed to read {md_filename}: {str(e)}")
            return

        base_name = md_filename.rsplit(".", 1)[0]
        image_count = 1
        updated = False

        while "<!-- image -->" in content:
            image_filename = f"{base_name}-figure-{image_count}.png"
            image_path = images_dir / image_filename

            if not image_path.exists():
                print(f"Image not found: {image_filename}")
                break

            caption = await self.get_image_caption(str(image_path))
            if caption:
                image_markdown = f"![{caption}](images/{image_filename})\n\n_{caption}_"
                content = content.replace("<!-- image -->", image_markdown, 1)
                print(f"Processed image {image_count} for {base_name}")
                updated = True
            else:
                print(f"Failed to process image {image_filename}")
                break

            image_count += 1

        if updated:
            try:
                md_path.write_text(content, encoding="utf-8")
            except Exception as e:
                print(f"Failed to write updated content to {md_filename}: {str(e)}")


async def main():
    args = parse_args()
    output_dir = Path(args.input_dir)

    if not output_dir.exists():
        print(f"Input directory not found: {output_dir}")
        return

    processor = DocumentProcessor(host=args.host, port=args.port)
    md_files = list(output_dir.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in {output_dir}")
        return

    for md_file in md_files:
        await processor.process_markdown_file(output_dir, md_file.name)

    print("Processing completed")


if __name__ == "__main__":
    asyncio.run(main())
