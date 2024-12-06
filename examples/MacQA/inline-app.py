# async def main():
#     os.environ["INFERENCE_MODEL"] = "meta-llama/Llama-3.2-1B-Instruct"
#     client = await LlamaStackDirectClient.from_template("ollama")
#     await client.initialize()
#     response = await client.models.list()
#     print(response)
#     model_name = response[0].identifier
#     response = await client.inference.chat_completion(
#         messages=[
#             SystemMessage(content="You are a friendly assistant.", role="system"),
#             UserMessage(
#                 content="hello world, write me a 2 sentence poem about the moon",
#                 role="user",
#             ),
#         ],
#         model_id=model_name,
#         stream=False,
#     )
#     print("\nChat completion response:")
#     print(response, type(response))


# asyncio.run(main())


import asyncio
import json
import os
import re
import uuid
from queue import Queue
from threading import Thread
from typing import AsyncGenerator, Generator, List, Optional

import gradio as gr
import requests
from dotenv import load_dotenv
from llama_stack.apis.memory_banks.memory_banks import VectorMemoryBankParams
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger

# pip install aiosqlite ollama faiss
from llama_stack_client.lib.direct.direct import LlamaStackDirectClient
from llama_stack_client.types import SystemMessage, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document


# Load environment variables
load_dotenv()

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "5000"))
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
USE_GPU = os.getenv("USE_GPU", False)
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
os.environ["INFERENCE_MODEL"] = MODEL_NAME
# if use_gpu, then the documents will be processed to output folder
DOCS_DIR = "/root/rag_data/output" if USE_GPU else "/root/rag_data/"

CUSTOM_CSS = """
.context-block {
    font-size: 0.8em;
    border-left: 3px solid #e9ecef;
    margin: 0.5em 0;
    padding: 0.5em 1em;
    opacity: 0.85;
}

.context-title {
    font-size: 0.8em;
    color: #9ca3af;
    font-weight: 400;
    display: flex;
    align-items: center;
    gap: 0.5em;
    margin-bottom: 0.3em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.context-title::before {
    content: "📄";
    font-size: 1em;
    opacity: 0.7;
}

.context-content {
    color: #6b7280;
    line-height: 1.4;
    font-weight: 400;
}

.inference-response {
    font-size: 1em;
    color: #111827;
    line-height: 1.5;
    margin-top: 1em;
}
"""
YAML = """
version: '2'
built_at: '2024-10-08T17:40:45.325529'
image_name: local
docker_image: null
conda_env: local
apis:
- shields
- agents
- models
- memory
- memory_banks
- inference
- safety
providers:
  inference:
  - provider_id: remote::ollama
    provider_type: remote::ollama
    config:
      url: http://127.0.0.1:11434
  memory:
  - provider_id: remote::chromadb
    provider_type: remote::chromadb
    config:
      host: localhost
      port: 6000
      protocol: http
  safety:
  - provider_id: inline::llama-guard-0
    provider_type: inline::llama-guard
    config:
      excluded_categories: []
  agents:
  - provider_id: inline::meta-reference-0
    provider_type: inline::meta-reference
    config:
      persistence_store:
        namespace: null
        type: sqlite
        db_path: ./runtime/kvstore.db
  telemetry:
  - provider_id: inline::meta-reference-0
    provider_type: inline::meta-reference
    config: {}
metadata_store: null
models:
- metadata: {}
  model_id: meta-llama/Llama-3.2-1B-Instruct
  provider_id: null
  provider_model_id: llama3.2:1b-instruct-fp16
shields: []
memory_banks: []
datasets: []
scoring_fns: []
eval_tasks: []
"""
YAML_PATH = os.getenv("YAML_PATH", "./llama_stack_run.yaml")


class LlamaChatInterface:
    def __init__(self, host: str, port: int, docs_dir: str):
        self.host = host
        self.port = port
        self.docs_dir = docs_dir
        self.client = None
        self.agent = None
        self.session_id = None
        self.memory_bank_id = "docqa_bank"

    def save_yaml(self):
        with open(YAML_PATH, "w") as f:
            f.write(YAML)
        print("YAML file saved.")

    async def initialize_system(self):
        """Initialize the entire system including memory bank and agent."""
        self.save_yaml()
        self.client = await LlamaStackDirectClient.from_config(YAML_PATH)
        await self.client.initialize()
        await self.setup_memory_bank()
        await self.initialize_agent()

    async def setup_memory_bank(self):
        """Set up the memory bank if it doesn't exist."""
        providers = await self.client.providers.list()
        provider_id = providers["memory"][0].provider_id
        memory_banks = await self.client.memory_banks.list()
        print(f"Memory banks: {memory_banks}")

        # Check if memory bank exists by identifier
        if any(bank.identifier == self.memory_bank_id for bank in memory_banks):
            print(f"Memory bank '{self.memory_bank_id}' exists.")
        else:
            print(f"Memory bank '{self.memory_bank_id}' does not exist. Creating...")
            await self.client.memory_banks.register(
                memory_bank_id=self.memory_bank_id,
                params=VectorMemoryBankParams(
                    embedding_model="all-MiniLM-L6-v2",
                    chunk_size_in_tokens=100,
                    overlap_size_in_tokens=10,
                ),
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
            await self.client.memory.insert(
                bank_id=self.memory_bank_id,
                documents=documents,
            )
            print(f"Loaded {len(documents)} documents from {self.docs_dir}")

    async def initialize_agent(self):
        """Initialize the agent with model registration and configuration."""

        agent_config = AgentConfig(
            model=MODEL_NAME,
            instructions="You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
            sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
            tools=[
                {
                    "type": "memory",
                    "memory_bank_configs": [
                        {"bank_id": self.memory_bank_id, "type": "vector"}
                    ],
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

        if self.agent is None:
            asyncio.run(self.initialize_system())

        response = self.agent.create_turn(
            messages=[{"role": "user", "content": message}],
            session_id=self.session_id,
        )

        current_response = ""
        context_shown = False

        for log in EventLogger().log(response):
            log.print()
            if hasattr(log, "content"):
                # Format context blocks if present
                if not context_shown and "Retrieved context from banks" in str(log):
                    context = self.format_context(str(log))
                    current_response = context + current_response
                    context_shown = True
                else:
                    current_response += log.content

                history[-1][1] = current_response
                yield history.copy()

    def format_context(self, log_str: str) -> str:
        """Format the context block with custom styling."""
        # Extract context and clean up the markers
        context_match = re.search(
            r"Retrieved context from banks:.*?\n(.*?===.*?===.*?)(?=\n>|$)",
            log_str,
            re.DOTALL,
        )
        if context_match:
            context = context_match.group(1).strip()
            # Remove the marker lines
            context = re.sub(
                r"====\s*Here are the retrieved documents for relevant context:\s*===\s*START-RETRIEVED-CONTEXT\s*===\s*",
                "",
                context,
                flags=re.IGNORECASE,
            )
            return f"""
<div class="context-block">
    <div class="context-title">Retrieved Context</div>
    <div class="context-content">{context}</div>
</div>
"""
        return ""


def create_gradio_interface(
    host: str = HOST,
    port: int = PORT,
    docs_dir: str = DOCS_DIR,
):
    chat_interface = LlamaChatInterface(host, port, docs_dir)

    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as interface:
        gr.Markdown("# LlamaStack Chat")

        chatbot = gr.Chatbot(
            bubble_full_width=False,
            show_label=False,
            height=400,
            container=True,
            render_markdown=True,
        )
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
    interface.launch(server_name=HOST, server_port=GRADIO_SERVER_PORT, debug=True)
