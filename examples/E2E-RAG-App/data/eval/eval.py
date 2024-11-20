import asyncio
import json
import os
import uuid
from typing import List, Optional

import chromadb
import fire
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datasets import Dataset
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
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
chroma_client = chromadb.PersistentClient(path="chroma")


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


def insert_documents_to_chromadb(file_dir: str, chunk_size: int = 350) -> None:
    """Inserts text documents from a directory into ChromaDB."""
    collection_name = "documents"
    existing_collections = chroma_client.list_collections()
    collection_names = [col.name for col in existing_collections]

    if collection_name in collection_names:
        cprint(
            f"Collection '{collection_name}' already exists. Skipping document insertion.",
            "yellow",
        )
        return

    collection = chroma_client.create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    cprint(f"Collection '{collection_name}' created.", "green")

    for filename in os.listdir(file_dir):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(file_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                chunks = chunk_text(content, chunk_size=chunk_size)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_chunk_{i}"
                    collection.add(
                        documents=[chunk],
                        ids=[chunk_id],
                        metadatas=[
                            {"filename": filename, "chunk_index": i, "content": chunk}
                        ],
                    )

    cprint(f"Inserted documents from {file_dir} into ChromaDB.", "green")


def query_chromadb(query: str) -> Optional[dict]:
    """Queries ChromaDB for relevant context based on input query."""
    collection = chroma_client.get_collection(
        name="documents", embedding_function=embedding_function
    )

    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    return results if results else None


async def get_response_with_context(
    agent: Agent, input_query: str, session_id: str
) -> (str, List[str]):
    """Fetches response from the agent with context from ChromaDB."""
    results = query_chromadb(input_query)
    if results and results["metadatas"]:
        context = "\n".join(
            f"Filename: {metadata['filename']}, Chunk index: {metadata['chunk_index']}\n{metadata['content']}"
            for metadata_list in results["metadatas"]
            for metadata in metadata_list
        )
        # Collect the contexts into a list
        contexts = [
            metadata["content"]
            for metadata_list in results["metadatas"]
            for metadata in metadata_list
        ]
    else:
        context = "No relevant context found."
        contexts = []

    messages = [
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {input_query}"}
    ]

    response = agent.create_turn(messages=messages, session_id=session_id)

    async for chunk in response:
        if chunk.event.payload.event_type == "turn_complete":
            return chunk.event.payload.turn.output_message.content, contexts

    return "No response generated.", contexts


async def run_main(host: str, port: int, docs_dir: str) -> None:
    """Main async function to register model, insert documents, and generate responses."""
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    insert_documents_to_chromadb(docs_dir)

    model_name = "Llama3.2-3B-Instruct"
    url = f"http://{host}:{port}/models/register"
    headers = {"Content-Type": "application/json"}
    data = {
        "model_id": model_name,
        "provider_model_id": None,
        "provider_id": "inline::meta-reference-0",
        "metadata": None,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    cprint(f"Model registration status: {response.status_code}", "blue")

    agent_config = AgentConfig(
        model=model_name,
        instructions="You are a helpful assistant that can answer questions based on provided documents.",
        sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
        enable_session_persistence=True,
    )
    agent = Agent(client, agent_config)

    # QA data
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
        {
            "Question": "What happens if an employee is absent without authorization for more than five consecutive working days?",
            "Answer": "If an employee is absent from their job without proper authorization for more than five consecutive working days, or fails to return from an approved leave, their absence will be deemed an 'automatic resignation.' (Source: Page 19, 'Automatic Resignation' section)",
        },
        {
            "Question": "How long is the normal probationary period for permanent civil service positions?",
            "Answer": "The document states that all appointments to permanent civil service positions require a probationary period, but the duration is governed by the collective bargaining agreement. Absences from work will extend the probationary period. (Source: Page 14, 'Probationary Period' section)",
        },
        {
            "Question": "What are employees required to do in case of a catastrophic event while off duty?",
            "Answer": "Employees should ensure the safety of their family and follow their department's instructions. If phone lines are down, they are required to listen to the radio for any reporting instructions. (Source: Page 51, 'Catastrophic Event While off Duty' section)",
        },
        {
            "Question": "What is the city's policy on accepting gifts from subordinates?",
            "Answer": "Employees may not solicit or accept any gifts from any subordinate, or any candidate or applicant for a position as an employee or subordinate to them. (Source: Page 49, 'Gifts' section)",
        },
    ]

    # Prepare lists to collect data
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
            generated_answer, contexts = await get_response_with_context(
                agent, question, session_id
            )
            cprint(f"Response: {generated_answer}", "green")

            # Append data to lists
            questions.append(question)
            generated_answers.append(generated_answer)
            retrieved_contexts.append(contexts)
            ground_truths.append(ground_truth_answer)
        except Exception as e:
            cprint(f"Error generating response for {question}: {e}", "red")

    # Create a Dataset for RAGAS
    eval_data = Dataset.from_dict(
        {
            "user_input": questions,
            "response": generated_answers,
            "retrieved_contexts": retrieved_contexts,
            "reference": ground_truths,
        }
    )

    # Run evaluation
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
    df.to_csv("evaluation_results2.csv", index=False)
    print(df.head())


def main(host: str, port: int, docs_dir: str) -> None:
    """Entry point for the script."""
    asyncio.run(run_main(host, port, docs_dir))


if __name__ == "__main__":
    fire.Fire(main)