import fire
from llama_stack import LlamaStack
from llama_stack.types import SamplingParams, UserMessage
from llama_stack.types.agentic_system_create_params import (
    AgentConfig,
    AgentConfigToolSearchToolDefinition,
    AgentConfigToolWolframAlphaToolDefinition,
)


def main(host: str, port: int):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    tool_definitions = [
        AgentConfigToolSearchToolDefinition(type="brave_search"),
        AgentConfigToolWolframAlphaToolDefinition(type="wolfram_alpha"),
    ]

    agentic_system_create_response = client.agentic_system.create(
        agent_config=AgentConfig(
            model="Meta-Llama3.1-8B-Instruct",
            instructions="You are a helpful assistant",
            tools=tool_definitions,
        )
    )
    print(agentic_system_create_response)

    agentic_system_create_session_response = client.agentic_system.sessions.create(
        agent_id=agentic_system_create_response.agent_id,
        session_name="test_session",
    )
    print(agentic_system_create_session_response)

    response = client.agentic_system.turns.create(
        agent_id=agentic_system_create_response.agent_id,
        session_id=agentic_system_create_session_response.session_id,
        messages=[
            UserMessage(content="What is the capital of France?", role="user"),
        ],
        stream=True,
    )

    for chunk in response:
        print(chunk)


if __name__ == "__main__":
    fire.Fire(main)
