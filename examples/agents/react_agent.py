# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# import os
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

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

REACT_PROMPT = """
You are an expert assistant who can solve any task using tool calls. You will be given a task to solve as best you can.
To do so, you have been given access to the following tools: <<tool_names>>
The way you use the tools is by specifying a json blob, ending with '<end_action>'.
Specifically, this json should have an `action` key (name of the tool to use) and an `action_input` key (input to the tool).

The $ACTION_JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. It should be formatted in json. Do not try to escape special characters. Here is the template of a valid $ACTION_JSON_BLOB:
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}<end_action>

Make sure to have the $INPUT as a dictionary in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

You should ALWAYS use the following format:

Thought: you should always think about one action to take. Then use the action as follows:
Action:
$ACTION_JSON_BLOB
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $ACTION_JSON_BLOB must only use a SINGLE action at a time.)

You can use the result of the previous action as input for the next action.
The observation will always be a string: it can represent a file, like "image_1.jpg".
Then you can use it as input for the next action. You can do it for instance as follows:

Observation: "image_1.jpg"

Thought: I need to transform the image that I received in the previous observation to make it green.
Action:
{
  "action": "image_transformer",
  "action_input": {"image": "image_1.jpg"}
}<end_action>

To provide the final answer to the task, use an action blob with "action": "final_answer" tool. It is the only way to complete the task, else you will be stuck on a loop. So your final output should look like this:
Action:
{
  "action": "final_answer",
  "action_input": {"answer": "insert your final answer here"}
}<end_action>


Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Thought: I will proceed step by step and use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.
Action:
{
  "action": "document_qa",
  "action_input": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}
}<end_action>

Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."


Thought: I will now generate an image showcasing the oldest person.
Action:
{
  "action": "image_generator",
  "action_input": {"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}
}<end_action>
Observation: "image.png"

Thought: I will now return the generated image.
Action:
{
  "action": "final_answer",
  "action_input": "image.png"
}<end_action>

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Thought: I will use python code evaluator to compute the result of the operation and then return the final answer using the `final_answer` tool
Action:
{
    "action": "python_interpreter",
    "action_input": {"code": "5 + 3 + 1294.678"}
}<end_action>
Observation: 1302.678

Thought: Now that I know the result, I will now return it.
Action:
{
  "action": "final_answer",
  "action_input": "1302.678"
}<end_action>

---
Task: "Which city has the highest population , Guangzhou or Shanghai?"

Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
Action:
{
    "action": "search",
    "action_input": "Population Guangzhou"
}<end_action>
Observation: ['Guangzhou has a population of 15 million inhabitants as of 2021.']


Thought: Now let's get the population of Shanghai using the tool 'search'.
Action:
{
    "action": "search",
    "action_input": "Population Shanghai"
}
Observation: '26 million (2019)'

Thought: Now I know that Shanghai has a larger population. Let's return the result.
Action:
{
  "action": "final_answer",
  "action_input": "Shanghai"
}<end_action>


Above example were using notional tools that might not exist for you. You only have access to these tools:
<<tool_descriptions>>

Here are the rules you should always follow to solve your task:
1. ALWAYS provide a single 'Thought:' sequence, and a single 'Action:' sequence that ends with <end_action>, else you will fail.
2. Always use the right arguments for the tools. Never use variable names in the 'action_input' field, use the value instead.
3. Call a tool only when needed: do not call the search agent if you do not need information, try to solve the task yourself.
4. Never re-do a tool call that you previously did with the exact same parameters.
5. Observations will be provided to you, no need to generate them

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""


class ReActOutputParser(OutputParser):
    def maybe_extract_action(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extract action name and parameters from the text format:

        Thought: <some_text>

        Action:
        {
        "action": <action_name>,
        "action_input": <action_params>
        }<end_action>

        Args:
            text (str): Input text containing the action block

        Returns:
            Tuple[str, Dict[str, Any]]: Tuple of (action_name, action_parameters)

        Raises:
            ValueError: If the action block cannot be parsed or is missing required fields
        """
        try:
            # Find the action block using regex
            action_pattern = r'Action:\s*{\s*"action":\s*"([^"]+)",\s*"action_input":\s*({[^}]+})\s*}<end_action>'
            match = re.search(action_pattern, text, re.DOTALL)

            if not match:
                raise ValueError("Could not find valid action block in text")

            action_name = match.group(1)
            action_params = json.loads(match.group(2))

            return action_name, action_params
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error parsing action: {e}")
            return None

    def parse(self, output_message: CompletionMessage) -> CompletionMessage:
        action = self._maybe_extract_action(output_message.content)
        if action is None:
            return output_message

        action_name, action_params = action
        call_id = str(uuid.uuid4())
        return CompletionMessage(
            content=output_message.content,
            tool_calls=[
                ToolCall(
                    call_id=call_id,
                    tool_name=action_name,
                    arguments=action_params,
                )
            ],
        )


class MetaExternalSearchTool(ClientTool):

    def get_name(self) -> str:
        return "get_external_meta_data"

    def get_description(self) -> str:
        return """
        Search the web for the given query about Meta. Get information Meta available on the public internet
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

    model = "meta-llama/Llama-3.3-70B-Instruct"

    client_tools = [
        MetaExternalSearchTool(),
    ]

    tool_names = ", ".join([tool.get_name() for tool in client_tools])
    tool_descriptions = "\n".join(
        [f"- {tool.get_name()}: {tool.get_description()}" for tool in client_tools]
    )
    instruction = REACT_PROMPT.replace("<<tool_names>>", tool_names).replace(
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
        tool_prompt_format="python_list",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
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

    # for chunk in response:
    #     pprint(chunk)


if __name__ == "__main__":
    fire.Fire(main)
