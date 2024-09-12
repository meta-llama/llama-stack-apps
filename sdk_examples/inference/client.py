import asyncio

import fire

from llama_stack import LlamaStack
from llama_stack.types import (
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    UserMessage,
)
from termcolor import cprint


class LogEvent:
    def __init__(
        self,
        content: str = "",
        end: str = "\n",
        color="white",
    ):
        self.content = content
        self.color = color
        self.end = "\n" if end is None else end

    def print(self, flush=True):
        cprint(f"{self.content}", color=self.color, end=self.end, flush=flush)


class EventLogger:
    async def log(self, event_generator):
        for chunk in event_generator:
            if isinstance(chunk, ChatCompletionStreamChunk):
                event = chunk.event
                if event.event_type == "start":
                    yield LogEvent("Assistant> ", color="cyan", end="")
                elif event.event_type == "progress":
                    yield LogEvent(event.delta, color="yellow", end="")
                elif event.event_type == "complete":
                    yield LogEvent("")
            elif isinstance(chunk, ChatCompletionResponse):
                yield LogEvent("Assistant> ", color="cyan", end="")
                yield LogEvent(chunk.completion_message.content, color="yellow")
            else:
                yield LogEvent("Assistant> ", color="cyan", end="")
                yield LogEvent(chunk, color="yellow")


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    message = UserMessage(
        content="hello world, troll me in two-paragraphs about 42", role="user"
    )
    cprint(f"User>{message.content}", "green")
    iterator = client.inference.chat_completion(
        messages=[
            UserMessage(
                content="hello world, troll me in two-paragraphs about 42", role="user"
            ),
        ],
        model="Meta-Llama3.1-8B-Instruct",
        stream=stream,
    )

    async for log in EventLogger().log(iterator):
        log.print()


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
