import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import chromadb
import fire
import requests
from datasets import Dataset
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    SemanticSimilarity,
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the evaluation script."""
    host: str
    port: int
    chroma_port: int
    model_name: str
    memory_bank_id: str
    docs_dir: Path

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        load_dotenv()
        return cls(
            host=os.getenv("HOST", "localhost"),
            port=int(os.getenv("PORT", 5000)),
            chroma_port=int(os.getenv("CHROMA_PORT", 8000)),
            model_name=os.getenv(
                "MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
            memory_bank_id=os.getenv("MEMORY_BANK_ID", "test_bank_236"),
            docs_dir=Path(os.getenv("DOCS_DIR", "../output")).resolve(),
        )


class MemoryBankManager:
    """Manages memory bank operations."""

    def __init__(self, client: LlamaStackClient, config: Config):
        self.client = client
        self.config = config

    async def setup(self) -> None:
        """Set up the memory bank if it doesn't exist."""
        try:
            providers = self.client.providers.list()
            provider_id = providers["memory"][0].provider_id
            memory_banks = self.client.memory_banks.list()

            if any(bank.identifier == self.config.memory_bank_id for bank in memory_banks):
                logger.info(
                    f"Memory bank '{self.config.memory_bank_id}' exists.")
                return

            logger.info(
                f"Creating memory bank '{self.config.memory_bank_id}'...")
            self.client.memory_banks.register(
                memory_bank_id=self.config.memory_bank_id,
                provider_id=provider_id,
            )
            await self._load_documents()
            logger.info("Memory bank registered successfully.")

        except Exception as e:
            logger.error(f"Failed to setup memory bank: {str(e)}")
            raise

    async def _load_documents(self) -> None:
        """Load documents from the specified directory into memory bank."""
        try:
            documents = []
            for file_path in self.config.docs_dir.glob("*.{txt,md}"):
                document = self._create_document(file_path)
                if document:
                    documents.append(document)

            if documents:
                self.client.memory.insert(
                    bank_id=self.config.memory_bank_id,
                    documents=documents,
                )
                logger.info(
                    f"Loaded {len(documents)} documents from {self.config.docs_dir}")

        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            raise

    def _create_document(self, file_path: Path) -> Optional[Document]:
        """Create a Document object from a file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return Document(
                document_id=file_path.name,
                content=content,
                mime_type="text/plain",
                metadata={"filename": file_path.name},
            )
        except Exception as e:
            logger.error(
                f"Failed to create document from {file_path}: {str(e)}")
            return None

    def query(self, query: str) -> Optional[Dict[str, Any]]:
        """Query memory bank for relevant context."""
        try:
            response = self.client.memory.query(
                bank_id=self.config.memory_bank_id,
                query=[query],
            )

            if response.chunks and response.scores:
                return {
                    "documents": [chunk.content for chunk in response.chunks],
                    "metadatas": [{"content": chunk.content} for chunk in response.chunks],
                    "distances": response.scores
                }
            return None

        except Exception as e:
            logger.error(f"Failed to query memory: {str(e)}")
            return None


class ResponseGenerator:
    """Handles generation of responses using the agent."""

    def __init__(self, agent: Agent, memory_manager: MemoryBankManager):
        self.agent = agent
        self.memory_manager = memory_manager

    async def get_response(self, query: str, session_id: str) -> Tuple[str, List[str]]:
        """Generate a response for the given query using context from memory."""
        try:
            context, contexts = self._get_context(query)
            messages = [
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}]

            response = self.agent.create_turn(
                messages=messages, session_id=session_id)
            return self._process_response(response), contexts

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return "Error generating response.", []

    def _get_context(self, query: str) -> Tuple[str, List[str]]:
        """Get context for the query from memory."""
        results = self.memory_manager.query(query)
        if results and results["metadatas"]:
            contexts = [metadata["content"]
                        for metadata in results["metadatas"]]
            context = "\n".join(f"Content:\n{ctx}" for ctx in contexts)
            return context, contexts
        return "No relevant context found.", []

    def _process_response(self, response) -> str:
        """Process the response from the agent."""
        full_response = ""
        for chunk in response:
            if hasattr(chunk, "event"):
                if chunk.event.payload.event_type == "turn_complete":
                    return chunk.event.payload.turn.output_message.content
                elif hasattr(chunk.event.payload, "delta"):
                    full_response += chunk.event.payload.delta.content or ""
        return full_response or "No response generated."


class Evaluator:
    """Handles evaluation of the question-answering system."""

    def __init__(self, config: Config):
        self.config = config

    async def run_evaluation(self) -> None:
        """Run the evaluation process."""
        try:
            client = self._setup_client()
            memory_manager = MemoryBankManager(client, self.config)
            await memory_manager.setup()

            agent = self._setup_agent(client)
            response_generator = ResponseGenerator(agent, memory_manager)

            qa_data = self._load_qa_data()
            results = await self._process_questions(response_generator, qa_data)
            self._evaluate_and_save_results(results)

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def _setup_client(self) -> LlamaStackClient:
        """Set up the LlamaStack client."""
        return LlamaStackClient(base_url=f"http://{self.config.host}:{self.config.port}")

    def _setup_agent(self, client: LlamaStackClient) -> Agent:
        """Set up the agent with configuration."""
        agent_config = AgentConfig(
            model=self.config.model_name,
            instructions="You are a helpful assistant that can answer questions based on provided documents.",
            sampling_params={"strategy": "greedy",
                             "temperature": 1.0, "top_p": 0.9},
            tools=[{
                "type": "memory",
                "memory_bank_configs": [{"bank_id": self.config.memory_bank_id, "type": "vector"}],
            }],
            tool_choice="auto",
            tool_prompt_format="json",
            enable_session_persistence=True,
        )
        return Agent(client, agent_config)

    def _load_qa_data(self) -> List[Dict[str, str]]:
        """Load QA evaluation data."""
        qa_file_path = Path(__file__).parent / "QA_eval.json"
        with qa_file_path.open('r') as f:
            return json.load(f)[:10]

    async def _process_questions(
        self,
        response_generator: ResponseGenerator,
        qa_data: List[Dict[str, str]]
    ) -> Dict[str, List]:
        """Process all questions and collect results."""
        results = {
            "questions": [],
            "generated_answers": [],
            "retrieved_contexts": [],
            "ground_truths": []
        }

        session_id = f"session-{uuid.uuid4()}"
        for qa in tqdm(qa_data, desc="Generating responses"):
            try:
                question = qa["Question"]
                ground_truth = qa["Answer"]

                answer, contexts = await response_generator.get_response(question, session_id)

                results["questions"].append(question)
                results["generated_answers"].append(answer)
                results["retrieved_contexts"].append(
                    [str(ctx) for ctx in contexts])
                results["ground_truths"].append(ground_truth)

            except Exception as e:
                logger.error(f"Failed to process question: {str(e)}")
                continue

        return results

    def _evaluate_and_save_results(self, results: Dict[str, List]) -> None:
        """Evaluate and save the results."""
        eval_data = Dataset.from_dict({
            "user_input": results["questions"],
            "response": results["generated_answers"],
            "retrieved_contexts": results["retrieved_contexts"],
            "reference": results["ground_truths"],
        })

        metrics = [
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
            FactualCorrectness(),
            SemanticSimilarity(),
        ]

        evaluation_result = evaluate(eval_data, metrics=metrics)
        df = evaluation_result.to_pandas()
        df.to_csv("evaluation_results.csv", index=False)
        logger.info("\nEvaluation Results:")
        logger.info("\n" + str(df))


async def run_main(docs_dir: str = None) -> None:
    """Main entry point for the evaluation script."""
    config = Config.from_env()
    if docs_dir:
        config.docs_dir = Path(docs_dir).resolve()

    evaluator = Evaluator(config)
    await evaluator.run_evaluation()


def main() -> None:
    """CLI entry point."""
    asyncio.run(run_main())


if __name__ == "__main__":
    fire.Fire(main)
