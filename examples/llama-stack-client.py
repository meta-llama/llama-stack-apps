# https://github.com/meta-llama/llama-stack-apps/blob/main/common/custom_tools.py

import json

from abc import abstractmethod
from typing import Dict, List

from llama_models.llama3.api.datatypes import *
from llama_stack.apis.agents import *

class CustomTool:
    """
    Developers can define their custom tools that models can use
    by extending this class.

    Developers need to provide
        - name
        - description
        - params_definition
        - implement tool's behavior in `run_impl` method

    NOTE: The return of the `run` method needs to be json serializable
    """

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        raise NotImplementedError

    def get_instruction_string(self) -> str:
        return f"Use the function '{self.get_name()}' to: {self.get_description()}"

    def parameters_for_system_prompt(self) -> str:
        return json.dumps(
            {
                "name": self.get_name(),
                "description": self.get_description(),
                "parameters": {
                    name: definition.__dict__
                    for name, definition in self.get_params_definition().items()
                },
            }
        )

    def get_tool_definition(self) -> FunctionCallToolDefinition:
        return FunctionCallToolDefinition(
            function_name=self.get_name(),
            description=self.get_description(),
            parameters=self.get_params_definition(),
        )

    @abstractmethod
    async def run(self, messages: List[Message]) -> List[Message]:
        raise NotImplementedError


class SingleMessageCustomTool(CustomTool):
    """
    Helper class to handle custom tools that take a single message
    Extending this class and implementing the `run_impl` method will
    allow for the tool be called by the model and the necessary plumbing.
    """

    async def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, "Expected single message"

        message = messages[0]

        tool_call = message.tool_calls[0]

        try:
            response = await self.run_impl(**tool_call.arguments)
            response_str = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            response_str = f"Error when running tool: {e}"

        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response_str,
        )
        return [message]

    @abstractmethod
    async def run_impl(self, *args, **kwargs):
        raise NotImplementedError()

# https://github.com/meta-llama/llama-stack-apps/blob/main/common/execute_with_custom_tools.py

from typing import AsyncGenerator, List

from llama_models.llama3.api.datatypes import *
from llama_stack.apis.agents import *
from llama_stack.apis.memory import *
from llama_stack.apis.safety import *

from llama_stack.apis.agents import AgentTurnResponseEventType as EventType

class AgentWithCustomToolExecutor:
    def __init__(
        self,
        api: Agents,
        agent_id: str,
        session_id: str,
        agent_config: AgentConfig,
        custom_tools: List[CustomTool],
    ):
        self.api = api
        self.agent_id = agent_id
        self.session_id = session_id
        self.agent_config = agent_config
        self.custom_tools = custom_tools

    async def execute_turn(
        self,
        messages: List[Message],
        attachments: Optional[List[Attachment]] = None,
        max_iters: int = 5,
        stream: bool = True,
    ) -> AsyncGenerator:
        tools_dict = {t.get_name(): t for t in self.custom_tools}

        current_messages = messages.copy()
        n_iter = 0
        while n_iter < max_iters:
            n_iter += 1

            request = AgentTurnCreateRequest(
                agent_id=self.agent_id,
                session_id=self.session_id,
                messages=current_messages,
                attachments=attachments,
                stream=stream,
            )

            turn = None
            async for chunk in self.api.create_agent_turn(request):
                if chunk.event.payload.event_type != EventType.turn_complete.value:
                    yield chunk
                else:
                    turn = chunk.event.payload.turn

            message = turn.output_message
            if len(message.tool_calls) == 0:
                yield chunk
                return

            if message.stop_reason == StopReason.out_of_tokens:
                yield chunk
                return

            tool_call = message.tool_calls[0]
            if tool_call.tool_name not in tools_dict:
                m = ToolResponseMessage(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    content=f"Unknown tool `{tool_call.tool_name}` was called. Try again with something else",
                )
                next_message = m
            else:
                tool = tools_dict[tool_call.tool_name]
                result_messages = await execute_custom_tool(tool, message)
                next_message = result_messages[0]

            yield next_message
            current_messages = [next_message]


async def execute_custom_tool(tool: CustomTool, message: Message) -> List[Message]:
    result_messages = await tool.run([message])
    assert (
        len(result_messages) == 1
    ), f"Expected single message, got {len(result_messages)}"

    return result_messages

# from https://github.com/meta-llama/llama-stack-apps/blob/main/common/client_utils.py

import os
import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from llama_models.llama3.api.datatypes import *
from llama_stack.apis.agents import *
from llama_stack.apis.agents.client import AgentsClient
from llama_stack.apis.memory import *
from llama_stack.apis.safety import *
from dotenv import load_dotenv


class AttachmentBehavior(Enum):
    rag = "rag"
    code_interpreter = "code_interpreter"
    auto = "auto"


# class ApiKeys(BaseModel):
#     wolfram_alpha: Optional[str] = None
#     brave: Optional[str] = None
#     bing: Optional[str] = None


# def load_api_keys_from_env() -> ApiKeys:
#     return ApiKeys(
#         bing=os.getenv("BING_SEARCH_API_KEY"),
#         brave=os.getenv("BRAVE_SEARCH_API_KEY"),
#         wolfram_alpha=os.getenv("WOLFRAM_ALPHA_API_KEY"),
#     )


# def search_tool_defn(api_keys: ApiKeys) -> SearchToolDefinition:
#     if not api_keys.brave and not api_keys.bing:
#         raise ValueError("You must specify either Brave or Bing search API key")

#     return SearchToolDefinition(
#         engine=SearchEngineType.bing if api_keys.bing else SearchEngineType.brave,
#         api_key=api_keys.bing if api_keys.bing else api_keys.brave,
#     )


# def default_builtins(api_keys: ApiKeys) -> List[ToolDefinitionCommon]:
#     return [
#         search_tool_defn(api_keys),
#         WolframAlphaToolDefinition(api_key=api_keys.wolfram_lpha),
#         PhotogenToolDefinition(),
#         CodeInterpreterToolDefinition(),
#     ]


class QuickToolConfig(BaseModel):
    custom_tools: List[CustomTool] = Field(default_factory=list)

    prompt_format: ToolPromptFormat = ToolPromptFormat.json

    # use this to control whether you want the model to write code to
    # process them, or you want to "RAG" them beforehand
    attachment_behavior: Optional[AttachmentBehavior] = None

    builtin_tools: List[ToolDefinitionCommon] = Field(default_factory=list)

    # if you have a memory bank already pre-populated, specify it here
    memory_bank_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


def enable_memory_tool(cfg: QuickToolConfig) -> bool:
    if cfg.memory_bank_id:
        return True
    return (
        cfg.attachment_behavior
        and cfg.attachment_behavior != AttachmentBehavior.code_interpreter
    )


# This is a utility function; it does not provide all bells and whistles
# you can get from the underlying Agents API. Any limitations should
# ideally be resolved by making another well-scoped utility function instead
# of adding complex options here.
async def make_agent_config_with_custom_tools(
    model: str = "Meta-Llama3.1-8B-Instruct",
    disable_safety: bool = False,
    tool_config: QuickToolConfig = QuickToolConfig(),
) -> AgentConfig:
    tool_definitions = []

    # ensure code interpreter is enabled if attachments need it
    builtin_tools = tool_config.builtin_tools
    tool_choice = ToolChoice.auto
    if tool_config.attachment_behavior == AttachmentBehavior.code_interpreter:
        if not any(isinstance(t, CodeInterpreterToolDefinition) for t in builtin_tools):
            builtin_tools.append(CodeInterpreterToolDefinition())

        tool_choice = ToolChoice.required

    tool_definitions = []

    if enable_memory_tool(tool_config):
        bank_configs = []
        if tool_config.memory_bank_id:
            bank_configs.append(
                AgentVectorMemoryBankConfig(bank_id=tool_config.memory_bank_id)
            )
        tool_definitions.append(MemoryToolDefinition(memory_bank_configs=bank_configs))

    tool_definitions += [t.get_tool_definition() for t in tool_config.custom_tools]

    if not disable_safety:
        for t in tool_definitions:
            t.input_shields = ["llama_guard"]
            t.output_shields = ["llama_guard", "injection_shield"]

    cfg = AgentConfig(
        model=model,
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(),
        tools=tool_definitions,
        tool_prompt_format=tool_config.prompt_format,
        tool_choice=tool_choice,
        input_shields=([] if disable_safety else ["llama_guard", "jailbreak_shield"]),
        output_shields=([] if disable_safety else ["llama_guard"]),
        enable_session_persistence=False,
    )
    return cfg


async def get_agent_with_custom_tools(
    base_url: str,
    agent_config: AgentConfig,
    custom_tools: List[CustomTool],
) -> AgentWithCustomToolExecutor:
    api = AgentsClient(base_url=base_url)

    create_response = await api.create_agent(agent_config)
    agent_id = create_response.agent_id

    name = f"Session-{uuid.uuid4()}"
    response = await api.create_agent_session(
        agent_id=agent_id,
        session_name=name,
    )
    session_id = response.session_id

    return AgentWithCustomToolExecutor(
        api, agent_id, session_id, agent_config, custom_tools
    )

# from https://github.com/meta-llama/llama-stack-apps/blob/main/common/event_logger.py

from typing import Optional

from llama_models.llama3.api.datatypes import *
from llama_models.llama3.api.tool_utils import ToolUtils

from llama_stack.apis.agents import AgentTurnResponseEventType, StepType

from termcolor import cprint


class LogEvent:
    def __init__(
        self,
        role: Optional[str] = None,
        content: str = "",
        end: str = "\n",
        color="white",
    ):
        self.role = role
        self.content = content
        self.color = color
        self.end = "\n" if end is None else end

    def __str__(self):
        if self.role is not None:
            return f"{self.role}> {self.content}"
        else:
            return f"{self.content}"

    def print(self, flush=True):
        #cprint(f"{str(self)}", color=self.color, end=self.end, flush=flush)
        print(f"{str(self)}", end=self.end, flush=flush)


EventType = AgentTurnResponseEventType


class EventLogger:
    async def log(
        self,
        event_generator,
        stream=True,
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
    ):
        previous_event_type = None
        previous_step_type = None

        async for chunk in event_generator:
            if not hasattr(chunk, "event"):
                # Need to check for custom tool first
                # since it does not produce event but instead
                # a Message
                if isinstance(chunk, ToolResponseMessage):
                    yield chunk, LogEvent(
                        role="CustomTool", content=chunk.content, color="grey"
                    )
                continue

            event = chunk.event
            event_type = event.payload.event_type
            if event_type in {
                EventType.turn_start.value,
                EventType.turn_complete.value,
            }:
                # Currently not logging any turn realted info
                yield event, None
                continue

            step_type = event.payload.step_type
            # handle safety
            if (
                step_type == StepType.shield_call
                and event_type == EventType.step_complete.value
            ):
                response = event.payload.step_details.response
                if not response.is_violation:
                    yield event, LogEvent(
                        role=step_type, content="No Violation", color="magenta"
                    )
                else:
                    yield event, LogEvent(
                        role=step_type,
                        content=f"{response.violation_type} {response.violation_return_message}",
                        color="red",
                    )

            # handle inference
            if step_type == StepType.inference:
                if stream:
                    if event_type == EventType.step_start.value:
                        # TODO: Currently this event is never received
                        yield event, LogEvent(
                            role=step_type, content="", end="", color="yellow"
                        )
                    elif event_type == EventType.step_progress.value:
                        # HACK: if previous was not step/event was not inference's step_progress
                        # this is the first time we are getting model inference response
                        # aka equivalent to step_start for inference. Hence,
                        # start with "Model>".
                        if (
                            previous_event_type != EventType.step_progress.value
                            and previous_step_type != StepType.inference
                        ):
                            yield event, LogEvent(
                                role=step_type, content="", end="", color="yellow"
                            )

                        if event.payload.tool_call_delta:
                            if isinstance(event.payload.tool_call_delta.content, str):
                                yield event, LogEvent(
                                    role=None,
                                    content=event.payload.tool_call_delta.content,
                                    end="",
                                    color="cyan",
                                )
                        else:
                            yield event, LogEvent(
                                role=None,
                                content=event.payload.model_response_text_delta,
                                end="",
                                color="yellow",
                            )
                    else:
                        # step_complete
                        yield event, LogEvent(role=None, content="")

                else:
                    # Not streaming
                    if event_type == EventType.step_complete.value:
                        response = event.payload.step_details.model_response
                        if response.tool_calls:
                            content = ToolUtils.encode_tool_call(
                                response.tool_calls[0], tool_prompt_format
                            )
                        else:
                            content = response.content
                        yield event, LogEvent(
                            role=step_type,
                            content=content,
                            color="yellow",
                        )

            # handle tool_execution
            if (
                step_type == StepType.tool_execution
                and
                # Only print tool calls and responses at the step_complete event
                event_type == EventType.step_complete.value
            ):
                details = event.payload.step_details
                for t in details.tool_calls:
                    yield event, LogEvent(
                        role=step_type,
                        content=f"Tool:{t.tool_name} Args:{t.arguments}",
                        color="green",
                    )
                for r in details.tool_responses:
                    yield event, LogEvent(
                        role=step_type,
                        content=f"Tool:{r.tool_name} Response:{r.content}",
                        color="green",
                    )

            if (
                step_type == StepType.memory_retrieval
                and event_type == EventType.step_complete.value
            ):
                details = event.payload.step_details
                content = interleaved_text_media_as_str(details.inserted_context)
                content = content[:200] + "..." if len(content) > 200 else content

                yield event, LogEvent(
                    role=step_type,
                    content=f"Retrieved context from banks: {details.memory_bank_ids}.\n====\n{content}\n>",
                    color="cyan",
                )

            preivous_event_type = event_type
            previous_step_type = step_type

# https://github.com/meta-llama/llama-stack-apps/blob/main/examples/scripts/multi_turn.py


import os
import sys

from typing import List, Optional

from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *
from llama_stack.apis.agents import *

class UserTurnInput(BaseModel):
    message: UserMessage
    attachments: Optional[List[Attachment]] = None


def prompt_to_turn(
    content: str, attachments: Optional[List[Attachment]] = None
) -> UserTurnInput:
    return UserTurnInput(message=UserMessage(content=content), attachments=attachments)


async def execute_turns(
    *,
    agent_config: AgentConfig,
    custom_tools: List[CustomTool],
    turn_inputs: List[UserTurnInput],
    base_url: str
):
    agent = await get_agent_with_custom_tools(
        base_url=base_url,
        agent_config=agent_config,
        custom_tools=custom_tools,
    )
    while len(turn_inputs) > 0:
        turn = turn_inputs.pop(0)

        iterator = agent.execute_turn(
            [turn.message],
            turn.attachments,
        )
        print(f"\nUser> {turn.message.content}\n") #, color="white", attrs=["bold"])

        async for event, log in EventLogger().log(iterator):
            if log is not None:
                log.print()                    
