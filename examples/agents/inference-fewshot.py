from llama_stack_client import LlamaStackClient
from llama_stack_client.types import CompletionMessage, UserMessage
from termcolor import cprint

# Initialize the LlamaStackClient with the base URL for inference endpoint
client = LlamaStackClient(base_url="http://localhost:5000")

# Invoke chat_completion with the few-shot example set
response = client.inference.chat_completion(
    messages=[
        UserMessage(content="Have shorter, spear-shaped ears.", role="user"),
        CompletionMessage(
            content="That's Alpaca!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Known for their calm nature and used as pack animals in mountainous regions.",
            role="user",
        ),
        CompletionMessage(
            content="That's Llama!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Has a straight, slender neck and is smaller in size compared to its relative.",
            role="user",
        ),
        CompletionMessage(
            content="That's Alpaca!",
            role="assistant",
            stop_reason="end_of_message",
            tool_calls=[],
        ),
        UserMessage(
            content="Generally taller and more robust, commonly seen as guard animals.",
            role="user",
        ),
    ],
    model="Llama3.2-11B-Vision-Instruct",
)

cprint(f"> Response: {response.completion_message.content}", "cyan")
