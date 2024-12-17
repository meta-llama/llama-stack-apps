# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import sys
import textwrap
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import (
    Attachment,
    MemoryToolDefinition,
    SamplingParams,
    SearchToolDefinition,
    UserMessage,
)
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from termcolor import colored

from .utils import data_url_from_file

load_dotenv()


class AgentChoice(Enum):
    WebSearch = "WebSearch"
    Memory = "Memory"


class AgentStore:
    def __init__(self, host, port) -> None:
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        available_models = [model.identifier for model in self.client.models.list()]
        if not available_models:
            print(colored("No available models. Exiting.", "red"))
            sys.exit(1)

        self.model = available_models[0]
        print(f"Using model: {self.model}")

        self.agents = {}
        self.sessions = {}
        self.first_turn = {}
        self.system_message = {}

    async def initialize_agents(self, bank_ids: List[str]) -> None:
        self.agents[AgentChoice.WebSearch] = await self.get_agent(
            agent_type=AgentChoice.WebSearch
        )
        self.create_session(AgentChoice.WebSearch)
        # Create a live bank that holds live context
        self.live_bank = self.create_live_bank()

        self.bank_ids = bank_ids
        self.agents[AgentChoice.Memory] = await self.get_agent(
            agent_type=AgentChoice.Memory,
            agent_params={"bank_ids": self.bank_ids + [self.live_bank]},
        )
        self.create_session(AgentChoice.Memory)

    def create_live_bank(self):
        self.live_bank = "live_bank"
        providers = self.client.providers.list()
        self.client.memory_banks.register(
            memory_bank_id=self.live_bank,
            params={
                "memory_bank_type": "vector",
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size_in_tokens": 512,
                "overlap_size_in_tokens": 64,
            },
            provider_id=providers["memory"][0].provider_id,
        )
        # FIXME: To avoid empty banks
        self.append_to_live_memory_bank(
            "This is a live bank. It holds live context for this chat"
        )
        return self.live_bank

    async def get_agent(
        self,
        agent_type: AgentChoice,
        agent_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        if agent_type == AgentChoice.WebSearch:
            if "BRAVE_SEARCH_API_KEY" not in os.environ:
                print(
                    colored(
                        "You must set the BRAVE_SEARCH_API_KEY environment variable to use this example.",
                        "red",
                    )
                )
                sys.exit(1)

            tools = [
                SearchToolDefinition(
                    type="brave_search",
                    engine="brave",
                    api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
                ),
                MemoryToolDefinition(
                    type="memory",
                    max_chunks=5,
                    max_tokens_in_context=2048,
                    memory_bank_configs=[],
                    query_generator_config={
                        "type": "default",
                        "sep": " ",
                    },
                ),
            ]
            user_instructions = textwrap.dedent(
                """
                You are an agent that can search the web (using brave_search) to answer user questions.

                Your task is to search the web to get the information related to the provided question.
                Ask clarifying questions if needed to figure out appropriate search query.
                Cite the top sources with corresponding urls.
                Once you make a relevant search query, summarize the results to answer in the following format:
                ```
                This is what I found on the web:
                {add answer here}

                Sources:
                {add sources with corresponding links}
                ```
                Do NOT add any other greetings or explanations. Just make a search call and answer in the appropriate format.
                """
            )
            agent_config = AgentConfig(
                model=self.model,
                instructions="",
                sampling_params=SamplingParams(
                    strategy="greedy", temperature=0.0, top_p=0.95
                ),
                tools=tools,
                enable_session_persistence=True,
            )
        elif agent_type == AgentChoice.Memory:
            bank_ids = agent_params.get("bank_ids", [])
            tools = [
                MemoryToolDefinition(
                    type="memory",
                    max_chunks=5,
                    max_tokens_in_context=2048,
                    memory_bank_configs=[
                        {
                            "type": "vector",
                            "bank_id": bank_id,
                        }
                        for bank_id in bank_ids
                    ],
                    query_generator_config={
                        "type": "default",
                        "sep": " ",
                    },
                ),
            ]
            user_instructions = ""
            agent_config = AgentConfig(
                model=self.model,
                instructions="",
                sampling_params=SamplingParams(
                    strategy="greedy", temperature=0.0, top_p=0.95
                ),
                tools=tools,
                enable_session_persistence=True,
            )

        response = self.client.agents.create(
            agent_config=agent_config,
        )

        agent_id = response.agent_id
        # Use self.first_turn to keep track of whether it's the first turn for each agent or not
        # This helps knowing whether to send the system message or not
        self.first_turn[agent_id] = True
        # Use self.system_message to keep track of the system message for each agent
        self.system_message[agent_id] = user_instructions

        return agent_id

    def create_session(self, agent_choice: str) -> str:
        agent_id = self.agents[agent_choice]
        self.first_turn[agent_id] = True
        response = self.client.agents.session.create(
            agent_id=agent_id,
            session_name=f"Session-{uuid.uuid4()}",
        )
        self.sessions[agent_choice] = response.session_id
        return self.sessions[agent_choice]

    async def build_index(self, file_dir: str) -> str:
        """Build a memory bank from a directory of pdf files."""

        # 1. create memory bank
        providers = self.client.providers.list()
        # create a memory bank
        self.client.memory_banks.register(
            memory_bank_id="memory_bank",
            params={
                "memory_bank_type": "vector",
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size_in_tokens": 512,
                "overlap_size_in_tokens": 64,
            },
            provider_id=providers["memory"][0].provider_id,
        )
        # cprint(f"Created bank: {json.dumps(bank, indent=4)}", color="green")

        # 2. load pdfs from directory as raw text
        paths = []
        for filename in os.listdir(file_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(file_dir, filename)
                paths.append(file_path)

        documents = [
            Document(
                document_id=os.path.basename(path),
                content=data_url_from_file(path),
            )
            for path in paths
        ]
        # insert some documents
        self.client.memory.insert(bank_id="memory_bank", documents=documents)

        return "memory_bank"

    async def chat(self, agent_choice, message, attachments) -> str:
        assert (
            agent_choice in self.agents
        ), f"Agent of type {agent_choice} not initialized"
        agent_id = self.agents[agent_choice]

        messages = []
        # If it's the first turn, send the system message along with the user message
        if self.first_turn[agent_id]:
            if self.system_message[agent_id]:
                messages.append(
                    UserMessage(content=self.system_message[agent_id], role="user")
                )
            self.first_turn[agent_id] = False

        session_id = self.sessions[agent_choice]
        atts = []
        if attachments is not None:
            for attachment in attachments:
                atts.append(
                    Attachment(
                        content=data_url_from_file(attachment),
                        # hardcoded for now since mimetype is inferred from data_url
                        mime_type="text/plain",
                    )
                )
        messages.append(UserMessage(role="user", content=message))
        generator = self.client.agents.turn.create(
            agent_id=self.agents[agent_choice],
            session_id=self.sessions[agent_choice],
            messages=messages,
            attachments=atts,
            stream=True,
        )
        for chunk in generator:
            event = chunk.event
            event_type = event.payload.event_type
            # FIXME: Use the correct event type
            if event_type == "turn_complete":
                turn = event.payload.turn

        inserted_context = ""
        for step in turn.steps:
            # FIXME: Update to use typed step types instead of strings
            if step.step_type == "memory_retrieval":
                inserted_context = step.inserted_context
            if step.step_type == "tool_execution":
                inserted_context = "\n".join([tr.content for tr in step.tool_responses])

        return turn.output_message.content, inserted_context

    def append_to_live_memory_bank(self, text: str) -> None:
        document = Document(
            document_id=uuid.uuid4().hex,
            content=text,
        )
        self.client.memory.insert(bank_id=self.live_bank, documents=[document])

    async def clear_live_bank(self) -> None:
        # FIXME: This is a hack, ideally we should
        # clear an existing bank instead of creating a new one
        self.live_bank = self.create_live_bank()
        self.agents[AgentChoice.Memory] = await self.get_agent(
            agent_type=AgentChoice.Memory,
            agent_params={"bank_ids": self.bank_ids + [self.live_bank]},
        )
        self.create_session(AgentChoice.Memory)
