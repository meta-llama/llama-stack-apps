import json
import os

from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.memory.api import *  # noqa: F403
from llama_toolchain.memory.client import MemoryClient
from llama_toolchain.memory.common.file_utils import data_url_from_file

from common.client_utils import *  # noqa: F403

from .tools import SearchAndBrowse


class AgentChoice(Enum):
    SimpleAgent = "SimpleAgent"
    AgentWithMemory = "AgentWithMemory"
    AgentWithSearchAndBrowse = "AgentWithSearchAndBrowse"


class AgentStore:
    def __init__(self, host, port, model) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.agents = {}

    async def initialize_agents(self, bank_ids: str) -> None:
        self.agents[AgentChoice.AgentWithMemory] = await self.get_agent_with_rag(
            bank_ids
        )
        self.agents[AgentChoice.SimpleAgent] = await self.get_agent(
            agent_type=AgentChoice.SimpleAgent
        )
        self.agents[AgentChoice.AgentWithSearchAndBrowse] = await self.get_agent(
            agent_type=AgentChoice.AgentWithSearchAndBrowse
        )

    async def get_agent_with_rag(self, bank_ids: List[str]):
        agent_config = AgentConfig(
            model=self.model,
            instructions="",
            sampling_params=SamplingParams(temperature=0.0, top_p=0.95),
            tools=[
                MemoryToolDefinition(
                    memory_bank_configs=[
                        AgenticSystemVectorMemoryBankConfig(bank_id=bank_id)
                        for bank_id in bank_ids
                    ],
                    query_generator_config=DefaultMemoryQueryGeneratorConfig(),
                ),
            ],
        )
        agent = await get_agent_with_custom_tools(
            self.host,
            self.port,
            agent_config,
            [],  # TODO: Add custom tools
        )

        return agent

    async def get_agent(self, agent_type: AgentChoice):
        if agent_type == AgentChoice.SimpleAgent:
            agent_config = AgentConfig(
                model=self.model,
                instructions="",
                sampling_params=SamplingParams(temperature=0.0, top_p=0.95),
                tools=[
                    SearchToolDefinition(engine=SearchEngineType.brave),
                    MemoryToolDefinition(
                        memory_bank_configs=[],
                        query_generator_config=DefaultMemoryQueryGeneratorConfig(),
                    ),
                ],
            )
            custom_tools = []
        elif agent_type == AgentChoice.AgentWithSearchAndBrowse:
            search_and_browse = SearchAndBrowse()
            agent_config = AgentConfig(
                model=self.model,
                instructions="",
                sampling_params=SamplingParams(temperature=0.0, top_p=0.95),
                tools=[search_and_browse.get_tool_definition()],
            )
            custom_tools = [search_and_browse]

        agent = await get_agent_with_custom_tools(
            self.host,
            self.port,
            agent_config,
            custom_tools,
        )

        return agent

    async def build_index(self, file_dir: str) -> str:
        """Build a memory bank from a directory of pdf files."""

        client = MemoryClient(f"http://{self.host}:{self.port}")

        # 1. create memory bank
        bank = await client.create_memory_bank(
            name="memory_bank",
            config=VectorMemoryBankConfig(
                bank_id="memory_bank",
                embedding_model="all-MiniLM-L6-v2",
                chunk_size_in_tokens=512,
                overlap_size_in_tokens=64,
            ),
        )
        print(f"Created bank: {json.dumps(bank.dict(), indent=4)}")

        # 2. load pdfs from directory as raw text
        paths = []
        for filename in os.listdir(file_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(file_dir, filename)
                paths.append(file_path)

        # 3. add raw text to memory bank
        documents = [
            MemoryBankDocument(
                document_id=os.path.basename(path),
                content=data_url_from_file(path),
            )
            for path in paths
        ]

        # insert some documents
        await client.insert_documents(bank_id=bank.bank_id, documents=documents)

        return bank.bank_id

    async def chat(self, agent_choice, message, attachments) -> str:
        assert (
            agent_choice in self.agents
        ), f"Agent of type {agent_choice} not initialized"
        agent = self.agents[agent_choice]
        atts = []
        if attachments is not None:
            for attachment in attachments:
                atts.append(
                    Attachment(
                        content=data_url_from_file(attachment),
                        mime_type="text/plain",
                    )
                )
        iterator = agent.execute_turn(
            messages=[UserMessage(content=message)],
            attachments=atts,
        )
        async for chunk in iterator:
            if not hasattr(chunk, "event"):
                continue
            event = chunk.event
            event_type = event.payload.event_type
            if event_type == AgenticSystemTurnResponseEventType.turn_complete.value:
                turn = event.payload.turn

        inserted_context = ""
        for step in turn.steps:
            if step.step_type == StepType.memory_retrieval.value:
                inserted_context = step.inserted_context
            if step.step_type == StepType.tool_execution.value:
                inserted_context = "\n".join([tr.content for tr in step.tool_responses])

        return turn.output_message.content, inserted_context

    async def append_to_memory_bank(self, bank_id: str, text: str) -> None:
        client = MemoryClient(f"http://{self.host}:{self.port}")
        document = MemoryBankDocument(
            document_id=uuid.uuid4().hex,
            content=text,
        )
        # insert some documents
        await client.insert_documents(bank_id=bank_id, documents=[document])
