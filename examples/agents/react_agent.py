# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# import os
import json
import uuid
from typing import Any, Dict, List, Optional, Union

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.client_tool import ClientTool
from llama_stack_client.lib.agents.output_parser import OutputParser

# from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.agents.turn import CompletionMessage
from llama_stack_client.types.shared.tool_call import ToolCall
from llama_stack_client.types.shared.tool_response_message import ToolResponseMessage
from llama_stack_client.types.shared.user_message import UserMessage
from llama_stack_client.types.tool_def_param import Parameter
from rich.pretty import pprint

REACT_JSON_PROMPT = """
You are an expert assistant who can solve any task using tool calls. You will be given a task to solve as best you can.
To do so, you have been given access to the following tools: <<tool_names>>

You must always respond in the following JSON format:
{
    "thought": $THOUGHT_PROCESS,
    "action": {
        "tool_name": $TOOL_NAME,
        "tool_params": $TOOL_PARAMS
    },
    "observation": $OBSERVATION,
    "answer": $ANSWER
}

Specifically, this json should have a `thought` key, a `action` key, and an `observation` key.

The `action` key should specify the $TOOL_NAME the name of the tool to use and the `tool_params` key should specify the parameters key as input to the tool.

Make sure to have the $TOOL_PARAMS as a dictionary in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You should always think about one action to take, and have the `thought` key contain your thought process about this action.
The `observation` key should contain the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The action key must only use a SINGLE tool at a time.)

You can use the result of the previous action as input for the next action.
The observation will always be a string: it can represent a file, like "image_1.jpg".
Then you can use it as input for the next action. You can do it for instance as follows:

{
    "observation": "image_1.jpg",
    "thought": "I need to transform the image that I received in the previous observation to make it green.",
    "action": {
        "tool_name": "image_transformer",
        "tool_params": {"image": "image_1.jpg"}
    },
    "answer": null
}


To provide the final answer to the task, use the `answer` key. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
{
    "observation": "your observation",
    "thought": "you thought process",
    "action": null,
    "answer": "insert your final answer here"
}

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

{
    "thought": "I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.",
    "action": {
        "tool_name": "document_qa",
        "tool_params": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}
    },
    "observation": "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.",
    "answer": null
}

{
    "thought": "I will now generate an image showcasing the oldest person.",
    "action": {
        "tool_name": "image_generator",
        "tool_params": {"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}
    },
    "observation": "image.png",
    "answer": null
}

{
    "thought": "I will now return the generated image.",
    "action": null,
    "answer": "image.png"
}

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

{
    "thought": "I will use python code evaluator to compute the result of the operation and then return the final answer using the `final_answer` tool",
    "action": {
        "tool_name": "python_interpreter",
        "tool_params": {"code": "5 + 3 + 1294.678"}
    },
    "observation": 1302.678,
    "answer": null
}

{
    "thought": "Now that I know the result, I will now return it.",
    "action": null,
    "observation": null,
    "answer": 1302.678
}

---
Task: "Which city has the highest population , Guangzhou or Shanghai?"

{
    "thought": "I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.",
    "action": {
        "tool_name": "search",
        "tool_params": {"query": "Population Guangzhou"}
    },
    "observation": ['Guangzhou has a population of 15 million inhabitants as of 2021.'],
    "answer": null
}

{
    "thought": "Now let's get the population of Shanghai using the tool 'search'.",
    "action": {
        "tool_name": "search",
        "tool_params": {"query": "Population Shanghai"}
    },
    "observation": "26 million (2019)",
    "answer": null
}

{
    "thought": "Now I know that Shanghai has a larger population. Let's return the result.",
    "action": null,
    "observation": null,
    "answer": "Shanghai"
}

Above example were using notional tools that might not exist for you. You only have access to these tools:
<<tool_descriptions>>

Here are the rules you should always follow to solve your task:
1. ALWAYS answer in the JSON format with keys "observation", "thought", "action", "answer", else you will fail.
2. Always use the right arguments for the tools. Never use variable names in the 'tool_params' field, use the value instead.
3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.
4. Never re-do a tool call that you previously did with the exact same parameters.
5. Observations will be provided to you, no need to generate them

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

from pydantic import BaseModel


class Action(BaseModel):
    tool_name: str
    tool_params: Dict[str, Any]


class ReActOutput(BaseModel):
    thought: str
    action: Optional[Action] = None
    observation: Optional[str] = None
    answer: Optional[str] = None


class ReActOutputParser(OutputParser):
    def parse(self, output_message: CompletionMessage) -> CompletionMessage:
        response_text = str(output_message.content)
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing action: {e}")
            return output_message

        if response_json.get("answer", None):
            return output_message

        if response_json.get("action", None):
            tool_name = response_json["action"].get("tool_name", None)
            tool_params = response_json["action"].get("tool_params", None)
            if tool_name and tool_params:
                call_id = str(uuid.uuid4())
                output_message.tool_calls = [
                    ToolCall(
                        call_id=call_id, tool_name=tool_name, arguments=tool_params
                    )
                ]

        return output_message


class SearchTool(ClientTool):

    def get_name(self) -> str:
        return "search"

    def get_description(self) -> str:
        return """
        Search the web for the given query.
        """

    def get_params_definition(self) -> Dict[str, Parameter]:
        return {
            "query": Parameter(
                name="query",
                parameter_type="str",
                description="The query to use for querying the internet",
                required=True,
            )
        }

    def run(
        self, messages: List[Union[UserMessage, ToolResponseMessage]]
    ) -> List[Union[UserMessage, ToolResponseMessage]]:
        from rich.pretty import pprint

        pprint(messages)
        print("run for MetaExternalSearchTool called")
        dummy_response = """
            torchtune is a PyTorch library for easily authoring, finetuning and experimenting with LLMs.

            torchtune provides:

            PyTorch implementations of popular LLMs from Llama, Gemma, Mistral, Phi, and Qwen model families
            Hackable training recipes for full finetuning, LoRA, QLoRA, DPO, PPO, QAT, knowledge distillation, and more
            Out-of-the-box memory efficiency, performance improvements, and scaling with the latest PyTorch APIs
            YAML configs for easily configuring training, evaluation, quantization or inference recipes
            Built-in support for many popular dataset formats and prompt templates
        """
        return [
            ToolResponseMessage(
                call_id="random-id",
                tool_name=self.get_name(),
                content=dummy_response,
                role="tool",
            )
        ]


def main():
    client = LlamaStackClient(
        base_url="http://localhost:8321",
    )

    model = "meta-llama/Llama-3.1-8B-Instruct"

    client_tools = [
        SearchTool(),
    ]

    tool_names = ", ".join([tool.get_name() for tool in client_tools])
    tool_descriptions = "\n".join(
        [f"- {tool.get_name()}: {tool.get_description()}" for tool in client_tools]
    )
    instruction = REACT_JSON_PROMPT.replace("<<tool_names>>", tool_names).replace(
        "<<tool_descriptions>>", tool_descriptions
    )

    agent_config = AgentConfig(
        model=model,
        instructions=instruction,
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        client_tools=[
            client_tool.get_tool_definition() for client_tool in client_tools
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
        # response_format={
        #     "type": "json_schema",
        #     "json_schema": ReActOutput.model_json_schema(),
        # },
    )

    agent = Agent(client, agent_config, client_tools, output_parser=ReActOutputParser())

    session_id = agent.create_session(f"ttest-session-{uuid.uuid4().hex}")

    response = agent.create_turn(
        messages=[
            {"role": "user", "content": "What model families does torchtune support?"}
        ],
        session_id=session_id,
        stream=False,
    )
    pprint(response)


if __name__ == "__main__":
    fire.Fire(main)
