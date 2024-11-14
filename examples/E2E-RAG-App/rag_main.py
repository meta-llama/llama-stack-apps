import asyncio
import json
import os
import uuid
from typing import List, Optional

import chromadb
import fire
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import cprint
from tqdm import tqdm

# Initialization
load_dotenv()
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma")


def chunk_text(content: str, chunk_size: int = 500) -> List[str]:
    """Splits content into chunks of approximately `chunk_size` characters."""
    chunks = []
    current_chunk = []

    for paragraph in content.split("\n\n"):
        if sum(len(p) for p in current_chunk) + len(paragraph) <= chunk_size:
            current_chunk.append(paragraph)
        else:
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = [paragraph]

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks


def insert_documents_to_chromadb(file_dir: str, chunk_size: int = 1000) -> None:
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
    cprint(f"Querying ChromaDB with: {query}", "cyan")
    collection = chroma_client.get_collection(
        name="documents", embedding_function=embedding_function
    )
    results = collection.query(query_texts=[query], n_results=5)
    return results if results else None


async def get_response_with_context(
    agent: Agent, input_query: str, session_id: str
) -> str:
    """Fetches response from the agent with context from ChromaDB."""
    results = query_chromadb(input_query)
    context = (
        "No relevant context found."
        if not results or not results["metadatas"][0]
        else "\n".join(metadata["content"] for metadata in results["metadatas"][0])
    )

    messages = [
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {input_query}"}
    ]
    cprint("Embedding retrieval completed. Sending these context to agent:", "cyan")
    cprint(context, "cyan")

    response = agent.create_turn(messages=messages, session_id=session_id)

    async for chunk in response:
        if chunk.event.payload.event_type == "turn_complete":
            cprint(f"----input_query------- {input_query}", "magenta")
            return chunk.event.payload.turn.output_message.content

    return "No response generated."


async def run_main(host: str, port: int, docs_dir: str) -> None:
    """Main async function to register model, insert documents, and generate responses."""
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    insert_documents_to_chromadb(docs_dir)

    model_name = "Llama3.2-3B-Instruct"
    url = "http://localhost:5000/models/register"
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

    user_prompts = [
        "What is the name of the llama model released on Oct 24, 2024?",
        "What about Llama 3.1 model, what is the release date for it?",
        "When was llama 3.3 released?",
    ]

    session_id = agent.create_session(f"session-{uuid.uuid4()}")
    for prompt in tqdm(user_prompts, desc="Generating responses"):
        cprint(f"Generating response for: {prompt}", "green")
        try:
            response = await get_response_with_context(agent, prompt, session_id)
            cprint(f"Response: {response}", "green")
        except Exception as e:
            cprint(f"Error generating response for {prompt}: {e}", "red")


def main(host: str, port: int, docs_dir: str) -> None:
    """Entry point for the script."""
    asyncio.run(run_main(host, port, docs_dir))


if __name__ == "__main__":
    fire.Fire(main)
