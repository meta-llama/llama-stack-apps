# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import fire
import os
import json
import base64
import mimetypes

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.event_logger import EventLogger

from llama_stack_client.types import SamplingParams, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from llama_stack_client.types.agent_create_params import *  # noqa: F403

from termcolor import cprint
from tqdm import tqdm

def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"
    return data_url


class Agent:
    def __init__(self, host: str, port: int):
        self.client = LlamaStackClient(
            base_url=f"http://{host}:{port}",
        )
        self.sessions = []

    def create_agent(self, agent_config: AgentConfig):
        agentic_system_create_response = self.client.agents.create(
            agent_config=agent_config,
        )
        self.agent_id = agentic_system_create_response.agent_id
        self.create_new_session()
    
    def create_new_session(self):
        agentic_system_create_session_response = self.client.agents.session.create(
            agent_id=self.agent_id,
            session_name=f"session-{len(self.sessions)}",
        )
        self.session_id = agentic_system_create_session_response.session_id
        self.sessions.append(self.session_id)

    async def execute_turn(self, content: str):
        response = self.client.agents.turn.create(
            agent_id=self.agent_id,
            session_id=self.session_id,
            messages=[
                UserMessage(content=content, role="user"),
            ],
            stream=True,
        )

        for chunk in response:
            if chunk.event.payload.event_type != "turn_complete":
                yield chunk

    async def execute_turn_non_streaming(self, content: str) -> str:
        # Temporary hack to get around Agents only have streaming
        response = self.client.agents.turns.create(
            agent_id=self.agent_id,
            session_id=self.session_id,
            messages=[
                UserMessage(content=content, role="user"),
            ],
            stream=False,
        )
        if not response.startswith("data:"):
            raise RuntimeError("Invalid response")

        split = response.split("data: ")
        for event in split:
            if len(event) < 1:
                continue
            event_json = json.loads(event.strip())
            payload = event_json["event"]["payload"]
            if payload["event_type"] == "turn_complete":
                return payload["turn"]["output_message"]

    async def build_index(self, file_dir: str) -> str:
        """Build a memory bank from a directory of pdf files"""
        # 1. create memory bank
        bank_id = "memory_bank"
        self.client.memory_banks.register(
            memory_bank={
                "identifier": bank_id,
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size_in_tokens": 512,
                "overlap_size_in_tokens": 64,
                "provider_id": "meta-reference",
            }
        )

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
                mime_type="application/pdf",
            )
            for path in paths
        ]

        # insert some documents
        self.client.memory.insert(bank_id=bank_id, documents=documents)

        return bank_id


async def get_rag_agent(host: str, port: int, file_dir: str) -> Agent:
    agent = Agent(host=host, port=port)
    
    # build index from pdf
    memory_bank_id = await agent.build_index(file_dir=file_dir)

    # build agent config
    tool_definitions = []
    bank_configs = []
    bank_configs.append(
        {
            "bank_id": memory_bank_id,
            "type": "vector",
        }
    )

    tool_definitions.append(
        AgentConfigToolMemoryToolDefinition(
                type="memory",
                memory_bank_configs=bank_configs,
                query_generator_config={
                    "type": "default",
                    "sep": " ",
                },
                max_tokens_in_context=4096,
                max_chunks=10,
        )
    )

    agent_config = AgentConfig(
        model="Llama3.1-8B-Instruct",
        instructions="You are a helpful assistant capable of answering questions about the contents of a document. You have access to a memory bank that contains the contents of the document. ",
        sampling_params=SamplingParams(strategy="greedy", temperature=1.0, top_p=0.9),
        tools=tool_definitions,
        tool_choices="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )
    
    agent.create_agent(agent_config)
    return agent
