import asyncio
import json

import fire

from llama_stack import LlamaStack
from llama_stack.types.memory_bank_insert_params import Document
from termcolor import cprint


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    print("hi")
    # create a memory bank
    bank = client.memory_banks.create(
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
    cprint(f"> /memory_banks/create: {bank}", "green")

    retrieved_bank = client.memory_banks.retrieve(
        bank_id=bank["bank_id"],
    )
    cprint(f"> /memory_banks/get: {retrieved_bank}", "blue")

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

    # insert some documents
    client.memory_banks.insert(
        bank_id=bank["bank_id"],
        documents=documents,
    )

    # query the documents
    response = client.memory_banks.query(
        bank_id=bank["bank_id"],
        query=[
            "How do I use lora",
        ],
    )
    for chunk, score in zip(response.chunks, response.scores):
        print(f"Score: {score}")
        print(f"Chunk:\n========\n{chunk}\n========\n")

    response = client.memory_banks.query(
        bank_id=bank["bank_id"],
        query=[
            "Tell me more about llama3 and torchtune",
        ],
    )
    for chunk, score in zip(response.chunks, response.scores):
        print(f"Score: {score}")
        print(f"Chunk:\n========\n{chunk}\n========\n")


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
