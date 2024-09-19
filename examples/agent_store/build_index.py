import asyncio

import fire
from termcolor import cprint

from examples.agent_store.api import AgentStore

MODEL = "Meta-Llama3.1-8B-Instruct"


async def build_index(host, port, file_dir):
    api = AgentStore(host, port, MODEL)
    bank_id = await api.build_index(file_dir)
    cprint(f"Successfully created bank: {bank_id}", color="green")


def main(host: str, port: int, file_dir: str):
    asyncio.run(build_index(host, port, file_dir))


if __name__ == "__main__":
    fire.Fire(main)
