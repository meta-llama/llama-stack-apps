import asyncio
import json
import os
import uuid
from typing import List, Optional

import fire
import requests
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from datasets import Dataset
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
import chromadb
from dotenv import load_dotenv


load_dotenv()
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 5000))

CHROMA_PORT = int(os.getenv("CHROMA_PORT", 6000))


chroma_client = chromadb.HttpClient(host=HOST, port=CHROMA_PORT )

async def load_documents(client, docs_dir, memory_bank_id):
    """Load documents from the specified directory into memory bank."""
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                document = Document(
                    document_id=filename,
                    content=content,
                    mime_type="text/plain",
                    metadata={"filename": filename},
                )
                documents.append(document)
    if documents:
        client.memory.insert(
            bank_id=memory_bank_id,
            documents=documents,
        )
        print(f"Loaded {len(documents)} documents from {docs_dir}")

async def setup_memory_bank(client, docs_dir, memory_bank_id):
    """Set up the memory bank if it doesn't exist."""
    providers = client.providers.list()
    provider_id = providers["memory"][0].provider_id
    collections = chroma_client.list_collections()


    if any(col.name == memory_bank_id for col in collections):
        print(f"The collection '{memory_bank_id}' exists.")
    else:
        print(
            f"The collection '{memory_bank_id}' does not exist. Creating the collection..."
        )
        client.memory_banks.register(
            memory_bank_id=memory_bank_id,
            params={
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size_in_tokens": 100,
                "overlap_size_in_tokens": 10,
            },
            provider_id=provider_id,
        )
        await load_documents(client, docs_dir, memory_bank_id)
        print(f"Memory bank registered.")

async def get_response_with_context(agent, input_query, session_id):
    response = agent.create_turn(
        messages=[{"role": "user", "content": input_query}], session_id=session_id
    )

    generated_answer = ""
    retrieved_contexts = []

    async for event in response:
        if event.event.payload.event_type == "token":
            generated_answer += event.event.payload.token

        elif event.event.payload.event_type == "tool_use":
            if event.event.payload.tool == "memory":
                result = event.event.payload.result
                if result and isinstance(result, dict) and 'documents' in result:
                    retrieved_docs = result['documents']
                    for doc in retrieved_docs:
                        retrieved_contexts.append(doc.get("content", ""))
                else:
                    print(f"Tool use result: {event.event.payload.result}")

        elif event.event.payload.event_type == "turn_complete":
            break

    return generated_answer, retrieved_contexts

async def run_main(host: str, port: int, docs_dir: str) -> None:
    """Main async function to register model, insert documents, and generate responses."""
    client = LlamaStackClient(base_url=f"http://{host}:{port}")
    memory_bank_id = "test_bank_113"

    await setup_memory_bank(client, docs_dir, memory_bank_id)

    model_name = "Llama3.2-1B-Instruct"

    agent_config = AgentConfig(
        model=model_name,
        instructions="You are a helpful assistant that can answer questions based on provided documents.",
        sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": memory_bank_id, "type": "vector"}],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 512,
                "max_chunks": 5,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        enable_session_persistence=True,
    )
    agent = Agent(client, agent_config)

    qa_data = [
        {
            "Question": "What is the policy regarding smoking in City offices?",
            "Answer": "Smoking is not permitted in City offices, or within 20 feet of entrances, exits, or operable windows of public buildings. (Source: Page 46, 'Smoke-Free Workplace' section)",
        },
    ]

    questions = []
    generated_answers = []
    retrieved_contexts_list = []
    ground_truths = []

    session_id = agent.create_session(f"session-{uuid.uuid4()}")
    for qa in tqdm(qa_data, desc="Generating responses"):
        question = qa["Question"]
        ground_truth_answer = qa["Answer"]

        cprint(f"Generating response for: {question}", "green")
        try:
            generated_answer, retrieved_contexts = await get_response_with_context(
                agent, question, session_id
            )
            cprint(f"Response: {generated_answer}", "green")
            questions.append(question)
            generated_answers.append(generated_answer)
            retrieved_contexts_list.append(retrieved_contexts)
            ground_truths.append(ground_truth_answer)
        except Exception as e:
            cprint(f"Error generating response for {question}: {e}", "red")

    eval_data = Dataset.from_dict(
        {
            "user_input": questions,
            "response": generated_answers,
            "retrieved_contexts": retrieved_contexts_list,
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
    df.to_csv("evaluation_results.csv", index=False)
    print(df.head())

def main(docs_dir: str) -> None:
    """Entry point for the script."""
    asyncio.run(run_main(HOST, PORT, docs_dir))

if __name__ == "__main__":
    fire.Fire(main)
