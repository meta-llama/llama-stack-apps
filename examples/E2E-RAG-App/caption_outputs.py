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
