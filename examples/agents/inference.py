from llama_stack_client import LlamaStackClient
from llama_stack_client.types import SystemMessage, UserMessage

client = LlamaStackClient(
    base_url="http://localhost:5000",
)

response = client.inference.chat_completion(
    messages=[
        SystemMessage(content="pretend you are a llama", role="system"),
        UserMessage(
            content="hello world, write me a 2 sentence poem about the moon",
            role="user",
        ),
    ],
    model="Llama3.2-11B-Vision-Instruct",
)

print(response.completion_message.content)
