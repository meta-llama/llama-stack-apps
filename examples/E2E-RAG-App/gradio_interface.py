import asyncio
import json
import os
import uuid
from typing import AsyncGenerator, Generator, List, Optional

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


class LlamaChatInterface:
    def __init__(self, host: str, port: int, chroma_port: int, docs_dir: str):
        self.host = host
        self.port = port
        self.docs_dir = docs_dir
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.chroma_client = chromadb.HttpClient(host=host, port={chroma_port})
        self.agent = None
        self.session_id = None
        self.memory_bank_id = "test_bank_666"

    async def initialize_system(self):
        """Initialize the entire system including memory bank and agent."""
        await self.setup_memory_bank()
        await self.load_documents()
        await self.initialize_agent()

    async def setup_memory_bank(self):
        """Set up the memory bank if it doesn't exist."""
        providers = self.client.providers.list()
        provider_id = providers["memory"][0].provider_id
        collections = chroma_client.list_collections()

        if any(col.name == memory_bank_id for col in collections):
            print(f"The collection '{memory_bank_id}' exists.")
        else:
            print(
                f"The collection '{memory_bank_id}' does not exist. Creating the collection..."
            )
            memory_bank = self.client.memory_banks.register(
                memory_bank_id=self.memory_bank_id,
                params={
                    "embedding_model": "all-MiniLM-L6-v2",
                    "chunk_size_in_tokens": 100,
                    "overlap_size_in_tokens": 10,
                },
                provider_id=provider_id,
            )
            print(f"Memory bank registered: {memory_bank}")

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
        model_name = "Llama3.2-3B-Instruct"

        agent_config = AgentConfig(
            model=model_name,
            instructions="You are a helpful assistant that can answer questions based on provided documents.",
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

    async def chat_stream(
        self, message: str, history: List[List[str]]
    ) -> AsyncGenerator[List[List[str]], None]:
        """Stream chat responses token by token with proper history handling."""
        if self.agent is None:
            await self.initialize_system()

        # Initialize history if None
        history = history or []

        # Add user message to history
        history.append([message, ""])

        # Get streaming response from agent
        response = self.agent.create_turn(
            messages=[{"role": "user", "content": message}], session_id=self.session_id
        )

        # Stream the response using EventLogger
        current_response = ""
        async for log in EventLogger().log(response):
            if hasattr(log, "content"):
                current_response += log.content
                history[-1][1] = current_response
                yield history


def create_gradio_interface(
    host: str = "localhost",
    port: int = 5000,
    chroma_port: int = 6000,
    docs_dir: str = "./docs",
):
    # Initialize the chat interface
    chat_interface = LlamaChatInterface(host, port, docs_dir, chroma_port)

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

        # Set up event handlers with streaming
        submit_event = msg.submit(
            fn=chat_interface.chat_stream,
            inputs=[msg, chatbot],
            outputs=chatbot,
            queue=True,
        ).then(
            fn=lambda: "",  # Clear textbox after sending
            outputs=msg,
        )

        submit_click = submit.click(
            fn=chat_interface.chat_stream,
            inputs=[msg, chatbot],
            outputs=chatbot,
            queue=True,
        ).then(
            fn=lambda: "",  # Clear textbox after sending
            outputs=msg,
        )

        clear.click(clear_chat, outputs=[chatbot, msg], queue=False)

        # Add keyboard shortcut for submit
        msg.submit(lambda: None, None, None, api_name=False)

    return interface


if __name__ == "__main__":
    # Create and launch the Gradio interface
    interface = create_gradio_interface(docs_dir="/root/rag_data")
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
