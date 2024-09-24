import asyncio

import fire

from llama_stack import LlamaStack
from llama_stack.lib.inference.event_logger import EventLogger
from llama_stack.types import UserMessage
from termcolor import cprint


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon", role="user"
    )
    cprint(f"User>{message.content}", "green")
    iterator = client.inference.chat_completion(
        messages=[
            UserMessage(
                content="hello world, write me a 2 sentence poem about the moon",
                role="user",
            ),
        ],
        model="Meta-Llama3.1-8B-Instruct",
        stream=stream,
    )

    async for log in EventLogger().log(iterator):
        log.print()

    # query models endpoint
    models_response = client.models.list()
    print(models_response)


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
