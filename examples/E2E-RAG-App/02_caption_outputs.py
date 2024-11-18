import os
import asyncio
import base64
import mimetypes
from llama_stack_client import LlamaStackClient

HOST = "localhost"
PORT = 5000

def encode_image_to_data_url(file_path: str) -> str:
    """Encode an image file to a data URL."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type of file: {file_path}")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"

async def get_image_caption(client: LlamaStackClient, image_path: str) -> str:
    """Get caption for an image using LlamaStack Vision API."""
    data_url = encode_image_to_data_url(image_path)

    message = {
        "role": "user",
        "content": [
            {"image": {"uri": data_url}},
            "This image comes from a scan inside a document, please provide a high level caption of what you see inside the image. Your caption will be used inside a RAG app so make sure its descriptive of the image and can be used in the relavant context"
        ]
    }

    response = await client.inference.chat_completion(
        messages=[message],
        model="Llama3.2-11B-Vision-Instruct",
        stream=False,
    )
    
    return response.choices[0].message.content

async def process_markdown_file(client: LlamaStackClient, output_dir: str, md_filename: str):
    """Process a single markdown file and replace image placeholders with captions."""
    md_path = os.path.join(output_dir, md_filename)
    images_dir = os.path.join(output_dir, 'images')
    
    print(f"Processing: {md_filename}")
    
    # Read the markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get the base filename without extension
    base_name = os.path.splitext(md_filename)[0]
    
    # Find all image placeholders
    image_count = 1
    
    while '<!-- image -->' in content:
        # Construct the expected image filename
        image_filename = f"{base_name}-figure-{image_count}.png"
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_filename}")
            break
            
        try:
            # Get caption from the Vision API
            caption = await get_image_caption(client, image_path)
            
            # Create markdown image with caption
            image_markdown = f"![{caption}](images/{image_filename})\n\n_{caption}_"
            
            # Replace the first occurrence of the placeholder
            content = content.replace('<!-- image -->', image_markdown, 1)
            
            print(f"Processed image {image_count} for {base_name}")
            
        except Exception as e:
            print(f"Error processing image {image_filename}: {str(e)}")
            break
            
        image_count += 1
    
    # Write the updated content back to the file
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)

async def main():
    # Define the output directory
    output_dir = os.path.join('DATA', 'output')
    
    # Initialize LlamaStack client
    client = LlamaStackClient(base_url=f"http://{HOST}:{PORT}")
    
    # Get all markdown files in the output directory
    md_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
    
    # Process each markdown file
    for md_file in md_files:
        await process_markdown_file(client, output_dir, md_file)

if __name__ == "__main__":
    asyncio.run(main())
