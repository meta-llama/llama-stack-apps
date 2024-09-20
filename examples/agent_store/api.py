import json
import os
from typing import Any, Dict

from .utils import data_url_from_file

from common.client_utils import *  # noqa: F403

from llama_stack import LlamaStack
from llama_stack.types import Attachment, SamplingParams, UserMessage
from llama_stack.types.agent_create_params import (
    AgentConfig,
    AgentConfigToolSearchToolDefinition,
    AgentConfigToolShield,
    AgentConfigToolShieldMemoryBankConfigVector,
)
from llama_stack.types.agents import AgentsTurnStreamChunk
from llama_stack.types.memory_bank_insert_params import Document, MemoryBankInsertParams


class AgentChoice(Enum):
    WebSearch = "WebSearch"
    Memory = "Memory"


class AgentStore:
    def __init__(self, host, port, model) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.client = LlamaStack(base_url=f"http://{host}:{port}")
        self.agents = {}
        self.sessions = {}

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
            agent_params={"bank_ids": self.bank_ids + [self.live_bank["bank_id"]]},
        )
        self.create_session(AgentChoice.Memory)

    def create_live_bank(self):
        self.live_bank = self.client.memory_banks.create(
            body={
                "name": "live_bank",
                "config": {
                    "bank_id": "live_bank",
                    "embedding_model": "dragon-roberta-query-2",
                    "chunk_size_in_tokens": 512,
                    "overlap_size_in_tokens": 64,
                },
            },
        )
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
            tools = [
                AgentConfigToolSearchToolDefinition(
                    type="brave_search",
                    engine="brave",
                    api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
                ),
                AgentConfigToolShield(
                    type="memory",
                    max_chunks=5,
                    max_tokens_in_context=2048,
                    memory_bank_configs=[],
                ),
            ]
            agent_config = AgentConfig(
                model=self.model,
                instructions="",
                sampling_params=SamplingParams(
                    strategy="greedy", temperature=0.0, top_p=0.95
                ),
                tools=tools,
            )
        elif agent_type == AgentChoice.Memory:
            bank_ids = agent_params.get("bank_ids", [])
            tools = [
                AgentConfigToolShield(
                    type="memory",
                    max_chunks=5,
                    max_tokens_in_context=2048,
                    memory_bank_configs=[
                        AgentConfigToolShieldMemoryBankConfigVector(
                            type="vector",
                            bank_id=bank_id,
                        )
                        for bank_id in bank_ids
                    ],
                ),
            ]
            agent_config = AgentConfig(
                model=self.model,
                instructions="",
                sampling_params=SamplingParams(
                    strategy="greedy", temperature=0.0, top_p=0.95
                ),
                tools=tools,
            )

        response = self.client.agents.create(
            agent_config=agent_config,
        )

        return response.agent_id

    def create_session(self, agent_choice: str) -> str:
        agent_id = self.agents[agent_choice]
        response = self.client.agents.sessions.create(
            agent_id=agent_id,
            session_name=f"Session-{uuid.uuid4()}",
        )
        self.sessions[agent_choice] = response.session_id
        return self.sessions[agent_choice]

    # async def build_index(self, file_dir: str) -> str:
    #     """Build a memory bank from a directory of pdf files."""

    #     client = MemoryClient(f"http://{self.host}:{self.port}")

    #     # 1. create memory bank
    #     bank = await client.create_memory_bank(
    #         name="memory_bank",
    #         config=VectorMemoryBankConfig(
    #             bank_id="memory_bank",
    #             embedding_model="all-MiniLM-L6-v2",
    #             chunk_size_in_tokens=512,
    #             overlap_size_in_tokens=64,
    #         ),
    #     )
    #     print(f"Created bank: {json.dumps(bank.dict(), indent=4)}")

    #     # 2. load pdfs from directory as raw text
    #     paths = []
    #     for filename in os.listdir(file_dir):
    #         if filename.endswith(".pdf"):
    #             file_path = os.path.join(file_dir, filename)
    #             paths.append(file_path)

    #     # 3. add raw text to memory bank
    #     documents = [
    #         MemoryBankDocument(
    #             document_id=os.path.basename(path),
    #             content=data_url_from_file(path),
    #         )
    #         for path in paths
    #     ]

    #     # insert some documents
    #     await client.insert_documents(bank_id=bank.bank_id, documents=documents)

    #     return bank.bank_id

    async def chat(self, agent_choice, message, attachments) -> str:
        assert (
            agent_choice in self.agents
        ), f"Agent of type {agent_choice} not initialized"
        agent_id = self.agents[agent_choice]
        session_id = self.sessions[agent_choice]
        atts = []
        # if attachments is not None:
        #     for attachment in attachments:
        #         atts.append(
        #             Attachment(
        #                 content=data_url_from_file(attachment),
        #                 # hardcoded for now since mimetype is inferred from data_url
        #                 mime_type="text/plain",
        #             )
        #         )
        generator = self.client.agents.turns.create(
            agent_id=self.agents[agent_choice],
            session_id=self.sessions[agent_choice],
            messages=[UserMessage(role="user", content=message)],
            # attachments=atts,
            stream=True,
        )
        for chunk in generator:
            if not isinstance(chunk, AgentsTurnStreamChunk):
                continue
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
        # print(f"Inserting to live bank : {self.live_bank['bank_id']}")
        self.client.memory_banks.insert(
            bank_id=self.live_bank["bank_id"], documents=[document]
        )

    async def clear_live_bank(self) -> None:
        # FIXME: This is a hack, ideally we should
        # clear an existing bank instead of creating a new one
        self.live_bank = self.create_live_bank()
        self.agents[AgentChoice.Memory] = await self.get_agent(
            agent_type=AgentChoice.Memory,
            agent_params={"bank_ids": self.bank_ids + [self.live_bank["bank_id"]]},
        )
        self.create_session(AgentChoice.Memory)
