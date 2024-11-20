import asyncio
import json
import os
import uuid
from typing import AsyncGenerator, Generator, List, Optional
from threading import Thread
from queue import Queue


import chromadb


import gradio as gr
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document


# Load environment variables
load_dotenv()

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 5000))
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 6000))
DOCS_DIR = os.getenv("DOCS_DIR", "/root/E2E-RAG-App/example_data/")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", 7861))

class LlamaChatInterface:
    def __init__(self, host: str, port: int, chroma_port: int, docs_dir: str):
        self.host = host
        self.port = port
        self.docs_dir = docs_dir
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.chroma_client = chromadb.HttpClient(host=host, port=chroma_port)
        self.agent = None
        self.session_id = None
        self.memory_bank_id = "test_bank_691"

    async def initialize_system(self):
        """Initialize the entire system including memory bank and agent."""
        await self.setup_memory_bank()
        await self.initialize_agent()

    async def setup_memory_bank(self):
        """Set up the memory bank if it doesn't exist."""
        providers = self.client.providers.list()
        provider_id = providers["memory"][0].provider_id
        collections = self.chroma_client.list_collections()

        if any(col.name == self.memory_bank_id for col in collections):
            print(f"The collection '{self.memory_bank_id}' exists.")
        else:
            print(
                f"The collection '{self.memory_bank_id}' does not exist. Creating the collection..."
            )
            self.client.memory_banks.register(
                memory_bank_id=self.memory_bank_id,
                params={
                    "embedding_model": "all-MiniLM-L6-v2",
                    "chunk_size_in_tokens": 100,
                    "overlap_size_in_tokens": 10,
                },
                provider_id=provider_id,
            )
            await self.load_documents()
            print(f"Memory bank registered.")

    async def load_documents(self):
        """Load documents from the specified directory into memory bank."""
        documents = []
        for filename in os.listdir(self.docs_dir):
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(self.docs_dir, filename)
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
            self.client.memory.insert(
                bank_id=self.memory_bank_id,
                documents=documents,
            )
            print(f"Loaded {len(documents)} documents from {self.docs_dir}")

    async def initialize_agent(self):
        """Initialize the agent with model registration and configuration."""
        model_name = "Llama3.2-1B-Instruct"

        agent_config = AgentConfig(
            model=model_name,
            instructions="You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
            sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
            tools=[
                {
                    "type": "memory",
                    "memory_bank_configs": [
                        {"bank_id": self.memory_bank_id, "type": "vector"}
                    ],
                    "query_generator_config": {"type": "default", "sep": " "},
                    "max_tokens_in_context": 300,
                    "max_chunks": 5,
                }
            ],
            tool_choice="auto",
            tool_prompt_format="json",
            enable_session_persistence=True,
        )
        self.agent = Agent(self.client, agent_config)
        self.session_id = self.agent.create_session(f"session-{uuid.uuid4()}")

    def chat_stream(
        self, message: str, history: List[List[str]]
    ) -> Generator[List[List[str]], None, None]:
        """Stream chat responses token by token with proper history handling."""

        history = history or []
        history.append([message, ""])

        output_queue = Queue()

        def run_async():
            async def async_process():
                if self.agent is None:
                    await self.initialize_system()

                response = self.agent.create_turn(
                    messages=[{"role": "user", "content": message}], session_id=self.session_id
                )

                current_response = ""
                async for log in EventLogger().log(response):
                    log.print()
                    if hasattr(log, "content"):
                        current_response += log.content
                        history[-1][1] = current_response
                        output_queue.put(history.copy())

                output_queue.put(None)

            asyncio.run(async_process())

        thread = Thread(target=run_async)
        thread.start()

        while True:
            item = output_queue.get()
            if item is None:
                break
            else:
                yield item

        thread.join()


def create_gradio_interface(
    host: str = HOST,
    port: int = PORT,
    chroma_port: int = CHROMA_PORT,
    docs_dir: str = DOCS_DIR,
):
    chat_interface = LlamaChatInterface(host, port, chroma_port, docs_dir)

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# LlamaStack Chat")

        chatbot = gr.Chatbot(bubble_full_width=False, show_label=False, height=400)
        msg = gr.Textbox(
            label="Message",
            placeholder="Type your message here...",
            show_label=False,
            container=False,
        )
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        gr.Examples(
            examples=[
                "What topics are covered in the documents?",
                "Can you summarize the main points?",
                "Tell me more about specific details in the text.",
            ],
            inputs=msg,
        )

        def clear_chat():
            return [], ""

        submit_event = msg.submit(
            fn=chat_interface.chat_stream,
            inputs=[msg, chatbot],
            outputs=chatbot,
            queue=True,
        ).then(
            fn=lambda: "",
            outputs=msg,
        )

        submit_click = submit.click(
            fn=chat_interface.chat_stream,
            inputs=[msg, chatbot],
            outputs=chatbot,
            queue=True,
        ).then(
            fn=lambda: "",
            outputs=msg,
        )

        clear.click(clear_chat, outputs=[chatbot, msg], queue=False)

        msg.submit(lambda: None, None, None, api_name=False)
        interface.load(fn=chat_interface.initialize_system)

    return interface


if __name__ == "__main__":
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch(server_name=HOST, server_port=GRADIO_SERVER_PORT, share=True, debug=True)
