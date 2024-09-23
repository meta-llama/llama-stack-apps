import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path

import fire

from llama_stack import LlamaStack
from llama_stack.types.memory_insert_params import Document
from termcolor import cprint


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    print("hi")
    # create a memory bank
    bank = client.memory.create(
        body={
            "name": "test_bank",
            "config": {
                "bank_id": "test_bank",
                "embedding_model": "dragon-roberta-query-2",
                "chunk_size_in_tokens": 512,
                "overlap_size_in_tokens": 64,
            },
        },
    )
    cprint(f"> /memory/create: {bank}", "green")

    retrieved_bank = client.memory.retrieve(
        bank_id=bank["bank_id"],
    )
    cprint(f"> /memory/get: {retrieved_bank}", "blue")

    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]

    documents = [
        Document(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]

    this_dir = os.path.dirname(__file__)
    files = [Path(this_dir).parent.parent / "CONTRIBUTING.md"]
    documents += [
        Document(
            document_id=f"num-{i}",
            content=data_url_from_file(path),
        )
        for i, path in enumerate(files)
    ]

    # insert some documents
    client.memory.insert(
        bank_id=bank["bank_id"],
        documents=documents,
    )

    # query the documents
    response = client.memory.query(
        bank_id=bank["bank_id"],
        query=[
            "How do I use lora",
        ],
    )
    for chunk, score in zip(response.chunks, response.scores):
        print(f"Score: {score}")
        print(f"Chunk:\n========\n{chunk}\n========\n")

    response = client.memory.query(
        bank_id=bank["bank_id"],
        query=[
            "Tell me more about llama3 and torchtune",
        ],
    )
    for chunk, score in zip(response.chunks, response.scores):
        print(f"Score: {score}")
        print(f"Chunk:\n========\n{chunk}\n========\n")

    response = client.memory.query(
        bank_id=bank["bank_id"],
        query=[
            "Tell me more about llama models",
        ],
    )
    for chunk, score in zip(response.chunks, response.scores):
        print(f"Score: {score}")
        print(f"Chunk:\n========\n{chunk}\n========\n")

    memory_banks_response = client.memory_banks.list()
    print(memory_banks_response)


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
