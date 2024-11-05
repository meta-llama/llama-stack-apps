import asyncio

import httpx
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from termcolor import cprint

local_client = LlamaStackClient(base_url="http://localhost:5000")
cloud_client = LlamaStackClient(base_url="http://localhost:5001")


async def select_client() -> LlamaStackClient:
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(f"{local_client.base_url}/health")
            if response.status_code == 200:
                cprint("Using local client.", "yellow")
                return local_client
    except httpx.RequestError:
        pass
    cprint("Local client unavailable. Switching to cloud client.", "yellow")
    return cloud_client


async def get_llama_response(stream: bool = True):
    client = await select_client()
    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon", role="user"
    )
    cprint(f"User> {message.content}", "green")

    response = client.inference.chat_completion(
        messages=[message],
        model="Llama3.2-11B-Vision-Instruct",
        stream=stream,
    )

    if not stream:
        cprint(f"> Response: {response}", "cyan")
    else:
        async for log in EventLogger().log(response):
            log.print()


asyncio.run(get_llama_response())
