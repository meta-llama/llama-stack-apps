import fire
from llama_stack import LlamaStack
from llama_stack.types import UserMessage


def main(host: str, port: int, stream: bool = True):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    response_iterator = client.inference.chat_completion(
        messages=[
            UserMessage(
                content="hello world, troll me in two-paragraphs about 42", role="user"
            ),
        ],
        model="Meta-Llama3.1-8B-Instruct",
        stream=stream,
    )

    if not stream:
        print(response_iterator)
        return

    for line in response_iterator:
        print(line)


if __name__ == "__main__":
    fire.Fire(main)
