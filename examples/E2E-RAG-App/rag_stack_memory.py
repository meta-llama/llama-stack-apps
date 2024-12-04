import asyncio
import json
import os
import uuid
from typing import List, Optional

import fire
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datasets import Dataset
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    SemanticSimilarity,
)
from termcolor import cprint
from tqdm import tqdm

# Initialization
load_dotenv()
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)


def chunk_text(content: str, chunk_size: int = 500) -> List[str]:
    """Splits content into chunks with overlap."""
    chunks = []
    current_chunk = []
    overlap = 100

    for paragraph in content.split("\n\n"):
        if sum(len(p) for p in current_chunk) + len(paragraph) <= chunk_size:
            current_chunk.append(paragraph)
        else:
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = (
                [current_chunk[-1], paragraph] if current_chunk else [paragraph]
            )

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks


async def insert_documents_to_memory_bank(client: LlamaStackClient, docs_dir: str):
    """Inserts text documents from a directory into a memory bank."""
    memory_bank_id = "test_bank"
    providers = client.providers.list()
    provider_id = providers["memory"][0].provider_id

    # Register a memory bank
    memory_bank = client.memory_banks.register(
        memory_bank_id=memory_bank_id,
        params={
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
        },
        provider_id=provider_id,
    )
    cprint(f"Memory bank registered: {memory_bank}", "green")

    # Prepare documents for insertion
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                chunks = chunk_text(content, chunk_size=350)

                for i, chunk in enumerate(chunks):
                    document = Document(
                        document_id=f"{filename}_chunk_{i}",
                        content=chunk,
                        mime_type="text/plain",
                        metadata={"filename": filename, "chunk_index": i},
                    )
                    documents.append(document)

    # Insert documents into the memory bank
    client.memory.insert(
        bank_id=memory_bank_id,
        documents=documents,
    )
    cprint(
        f"Inserted documents from {docs_dir} into memory bank '{memory_bank_id}'.",
        "green",
    )


async def get_response_with_memory_bank(
    agent: Agent, input_query: str, session_id: str
) -> (str, List[str]):
    """Fetches response from the agent with context from the memory bank."""
    response = agent.create_turn(
        messages=[{"role": "user", "content": input_query}],
        session_id=session_id,
    )

    context_responses = []
    async for log in EventLogger().log(response):
        # Log the structure for debugging
        print(f"Log structure: {vars(log)}")

        # Ensure attribute existence before accessing
        if hasattr(log, "event") and hasattr(log.event, "payload"):
            if log.event.payload.event_type == "turn_complete":
                return log.event.payload.turn.output_message.content, context_responses
        else:
            print("Warning: The 'event' attribute or 'payload' is not present.")

    return "No response generated.", context_responses


async def run_main(host: str, port: int, docs_dir: str) -> None:
    """Main async function to register model, insert documents, and generate responses."""
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    # Insert documents to the memory bank
    await insert_documents_to_memory_bank(client, docs_dir)

    # Model registration
    model_name = "Llama3.2-3B-Instruct"
    response = requests.post(
        f"http://{host}:{port}/models/register",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "model_id": model_name,
                "provider_model_id": None,
                "provider_id": "inline::meta-reference-0",
                "metadata": None,
            }
        ),
    )
    cprint(f"Model registration status: {response.status_code}", "blue")

    # Agent configuration
    agent_config = AgentConfig(
        model=model_name,
        instructions="You are a helpful assistant that can answer questions based on provided documents.",
        sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": "test_bank", "type": "vector"}],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 4096,
                "max_chunks": 10,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        enable_session_persistence=True,
    )
    agent = Agent(client, agent_config)

    # QA data for testing
    qa_data = [
        {
            "Question": "What is the policy regarding smoking in City offices?",
            "Answer": "Smoking is not permitted in City offices, or within 20 feet of entrances, exits, or operable windows of public buildings. (Source: Page 46, 'Smoke-Free Workplace' section)",
        },
        {
            "Question": "How many days of paid sick leave do most full-time employees earn per year under Civil Service Rules?",
            "Answer": "Most full-time employees earn 13 8-hour working days per year of paid sick leave under the Civil Service Rules. (Source: Page 32, 'Accrual of Paid Sick Leave' section)",
        },
        {
            "Question": "What are the three categories of employees eligible for health coverage?",
            "Answer": "The following employees are eligible:\n\nAll permanent employees working at least 20 hours per week\n\nAll regularly scheduled provisional employees working at least 20 hours per week\n\nAll other employees (including temporary exempt or 'as needed') who have worked more than 1040 hours in any consecutive 12-month period and work at least 20 hours per week (Source: Page 25, 'Eligibility' section)",
        },
        {
            "Question": "How long must an employee wait before using vacation time after starting employment?",
            "Answer": "Employees are not eligible to use vacation in the first year of continuous service. After one year of continuous service, they are awarded vacation allowance at the rate of .0385 of an hour for each hour of paid service in the preceding year. (Source: Page 30, 'Vacation' section)",
        },
        {
            "Question": "What must an employee do if they're summoned for jury duty?",
            "Answer": "An employee must notify their supervisor as soon as they receive a jury summons. If required to report during working hours, they will be excused from work on the day they perform jury service, provided they give prior notification. If not selected or dismissed early, they must return to work as soon as possible. (Source: Page 37, 'Jury Duty Leave' section)",
        },
        # Add more questions as needed
    ]

    questions = []
    generated_answers = []
    retrieved_contexts = []
    ground_truths = []

    session_id = agent.create_session(f"session-{uuid.uuid4()}")
    for qa in tqdm(qa_data, desc="Generating responses"):
        question = qa["Question"]
        ground_truth_answer = qa["Answer"]

        cprint(f"Generating response for: {question}", "green")
        try:
            generated_answer, contexts = await get_response_with_memory_bank(
                agent, question, session_id
            )
            cprint(f"Response: {generated_answer}", "green")

            questions.append(question)
            generated_answers.append(generated_answer)
            retrieved_contexts.append(contexts)
            ground_truths.append(ground_truth_answer)
        except Exception as e:
            cprint(f"Error generating response for {question}: {e}", "red")

    # Create a Dataset for RAGAS evaluation
    eval_data = Dataset.from_dict(
        {
            "user_input": questions,
            "response": generated_answers,
            "retrieved_contexts": retrieved_contexts,
            "reference": ground_truths,
        }
    )

    result = evaluate(
        eval_data,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
            FactualCorrectness(),
            SemanticSimilarity(),
        ],
    )

    df = result.to_pandas()
    df.to_csv("evaluation_results_with_memory.csv", index=False)
    print(df.head())


def main(host: str, port: int, docs_dir: str) -> None:
    """Entry point for the script."""
    asyncio.run(run_main(host, port, docs_dir))


if __name__ == "__main__":
    fire.Fire(main)
