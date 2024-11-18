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
