import os
import asyncio
import base64
import mimetypes
from pathlib import Path
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger

HOST = "localhost"
PORT = 5000

def encode_image_to_data_url(file_path: str) -> str:
    """Encode an image file to a data URL."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type of the file")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"

class DocumentProcessor:
    def __init__(self):
        self.client = LlamaStackClient(base_url=f"http://{HOST}:{PORT}")
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
                    {"image": {"uri": data_url}},
                    "This image comes from a scan inside a document, please provide a high level caption of what you see inside the image."
                ]
            }

            response = await self.client.inference.chat_completion(
                messages=[message],
                model="Llama3.2-11B-Vision-Instruct",
                stream=False
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
        images_dir = Path(output_dir) / 'images'
        
        try:
            content = md_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Failed to read {md_filename}: {str(e)}")
            return

        base_name = md_filename.rsplit('.', 1)[0]
        image_count = 1
        updated = False

        while '<!-- image -->' in content:
            image_filename = f"{base_name}-figure-{image_count}.png"
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                print(f"Image not found: {image_filename}")
                break
            
            caption = await self.get_image_caption(str(image_path))
            if caption:
                image_markdown = f"![{caption}](images/{image_filename})\n\n_{caption}_"
                content = content.replace('<!-- image -->', image_markdown, 1)
                print(f"Processed image {image_count} for {base_name}")
                updated = True
            else:
                print(f"Failed to process image {image_filename}")
                break
            
            image_count += 1

        if updated:
            try:
                md_path.write_text(content, encoding='utf-8')
            except Exception as e:
                print(f"Failed to write updated content to {md_filename}: {str(e)}")

async def main():
    output_dir = Path('DATA') / 'output'
    
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    processor = DocumentProcessor()
    md_files = list(output_dir.glob('*.md'))
    
    if not md_files:
        print(f"No markdown files found in {output_dir}")
        return

    for md_file in md_files:
        await processor.process_markdown_file(output_dir, md_file.name)

    print("Processing completed")

if __name__ == "__main__":
    asyncio.run(main())
