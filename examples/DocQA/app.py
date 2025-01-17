import os
import re
import subprocess
from multiprocessing import freeze_support
from typing import Generator, List, Optional

import gradio as gr
from dotenv import load_dotenv

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document

# Load environment variables
load_dotenv()

GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
USE_DOCKER = os.getenv("USE_DOCKER", False)
USE_GPU_FOR_DOC_INGESTION = os.getenv("USE_GPU_FOR_DOC_INGESTION", False)
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
# if USE_GPU_FOR_DOC_INGESTION, then the documents will be processed to output folder
DOCS_DIR = "/root/rag_data/output" if USE_GPU_FOR_DOC_INGESTION else "/root/rag_data/"
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


class LlamaChatInterface:
    def __init__(self):
        self.docs_dir = None
        self.client = None
        self.agent = None
        self.session_id = None
        self.memory_bank_id = "doc_bank"

    def initialize_system(self, provider_name="ollama"):
        """Initialize the entire system including memory bank and agent."""
        # path_to_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "llama_stack_run.yaml"))
        self.client = LlamaStackAsLibraryClient(provider_name)
        #print(type(self.client.async_client.config), self.client.async_client.config)

        # Disable scoring and eval by modifying the config
        self.client.async_client.config.apis = [
            "agents",
            "datasetio",
            "inference",
            "memory",
            "safety",
            "telemetry",
            "tool_runtime",
        ]
        #self.client.async_client.config.tool_groups = []
        del self.client.async_client.config.providers["scoring"]
        del self.client.async_client.config.providers["eval"]

        # only enable memory_runtime
        tool_runtime = [] 
        for provider in self.client.async_client.config.providers["tool_runtime"]:
            if provider.provider_id == 'memory-runtime':
                tool_runtime.append(provider)
        self.client.async_client.config.providers["tool_runtime"] = tool_runtime
        # print(
        #     111, type(self.client.async_client.config), self.client.async_client.config
        # )

        self.client.initialize()
        self.docs_dir = DOCS_DIR
        self.setup_memory_bank()
        self.initialize_agent()

    def setup_memory_bank(self):
        """Set up the memory bank if it doesn't exist."""
        providers = self.client.providers.list()
        memory_provider = [provider for provider in providers if provider.api == "memory"]
        provider_id = memory_provider[0].provider_id
        memory_banks = self.client.memory_banks.list()
        print("Memory banks: ", memory_banks)
        # Check if memory bank exists by identifier
        if memory_banks and any(bank.identifier == self.memory_bank_id for bank in memory_banks):
            print(f"Memory bank '{self.memory_bank_id}' exists.")
        else:
            print(f"Memory bank '{self.memory_bank_id}' does not exist. Creating...")
            self.client.memory_banks.register(
                memory_bank_id=self.memory_bank_id,
                params={
                    "memory_bank_type": "vector",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "chunk_size_in_tokens": 200,
                    "overlap_size_in_tokens": 20,
                },
                provider_id=provider_id,
            )
            self.load_documents()
            print(f"Memory bank registered.")

    def load_documents(self):
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

    def initialize_agent(self):
        """Initialize the agent with model registration and configuration."""

        agent_config = AgentConfig(
            model=MODEL_NAME,
            instructions="You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
            toolgroups = [
            {
                "name": "builtin::memory",
                "args" : {"memory_bank_ids": [self.memory_bank_id]} 
                }
            ],
            enable_session_persistence=True,
        )
        self.agent = Agent(self.client, agent_config)
        self.session_id = self.agent.create_session(f"session-docqa")

    def chat_stream(
        self, message: str, history: List[List[str]]
    ) -> Generator[List[List[str]], None, None]:
        """Stream chat responses token by token with proper history handling."""
        try:
            history = history or []
            history.append([message, ""])

            response = self.agent.create_turn(
                messages=[{"role": "user", "content": message}],
                session_id=self.session_id,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise gr.exceptions.Error(e)
            return (f"Error: {e}",)

        current_response = ""

        for log in EventLogger().log(response):
            log.print()
            if hasattr(log, "content"):
                # Format context blocks if present
                if "tool_execution>" not in str(log):
                    current_response += log.content

                history[-1][1] = current_response
                yield history.copy()

    


def create_gradio_interface(
    docs_dir: str = DOCS_DIR,
):
    chat_interface = LlamaChatInterface()
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as main_interface:
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

    # Combine both interfaces
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        if not USE_DOCKER:
            with gr.Tab("Setup", visible=True) as setup_tab:
                with gr.Blocks(
                    theme=gr.themes.Soft(), css=CUSTOM_CSS
                ) as initial_interface:
                    gr.Markdown(
                        "## Use the following options to setup the chat interface"
                    )
                    folder_path_input = gr.Textbox(label="Data Folder Path")
                    # folder_path_input = gr.File(label="Select Data Directory for RAG", file_count="directory")
                    model_name_input = gr.Dropdown(
                        choices=[
                            "meta-llama/Llama-3.2-1B-Instruct",
                            "meta-llama/Llama-3.2-3B-Instruct",
                            "meta-llama/Llama-3.1-8B-Instruct",
                            "meta-llama/Llama-3.1-70B-Instruct",
                            "meta-llama/Llama-3.1-405B-Instruct",
                        ],
                        value="meta-llama/Llama-3.2-1B-Instruct",
                        label="Llama Model Name",
                    )
                    provider_name = gr.Dropdown(
                        choices=[
                            "ollama",
                            "together",
                            "fireworks",
                        ],
                        value="ollama",
                        label="Provider List",
                    )
                    api_key = gr.Textbox(
                        label="Put your API key here if you choose together or fireworks provider"
                    )
                    setup_button = gr.Button("Setup Chat Interface")
                    setup_output = gr.Textbox(label="Setup", interactive=False)

                    # Function to handle the initial input and transition to the chat interface
                    def setup_chat_interface(
                        folder_path, model_name, provider_name, api_key
                    ):
                        global MODEL_NAME
                        global DOCS_DIR
                        DOCS_DIR = folder_path
                        MODEL_NAME = model_name
                        if provider_name == "ollama":
                            ollama_name_dict = {
                                "meta-llama/Llama-3.2-1B-Instruct": "llama3.2:1b-instruct-fp16",
                                "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b-instruct-fp16",
                                "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b-instruct-fp16",
                            }
                            if model_name not in ollama_name_dict:
                                raise gr.exceptions.Error(
                                    f"Model {model_name} is not supported currently, please use 1B, 3B and 8B model."
                                )
                            else:
                                ollama_name = ollama_name_dict[model_name]
                            try:
                                print("Starting Ollama server...")
                                subprocess.Popen(
                                    f"/usr/local/bin/ollama run {ollama_name} --keepalive=99h".split(),
                                    stdout=subprocess.DEVNULL,
                                )
                                subprocess.run(["sleep", "3"], capture_output=True)
                            except Exception as e:
                                print(f"Error: {e}")
                                raise gr.exceptions.Error(e)
                                return (f"Error: {e}",)
                        elif provider_name == "together":
                            os.environ["TOGETHER_API_KEY"] = api_key
                        elif provider_name == "fireworks":
                            os.environ["FIREWORKS_API_KEY"] = api_key
                        try:
                            print("Starting LlamaStack direct client...")
                            os.environ["INFERENCE_MODEL"] = model_name
                            print("Using model: ", model_name)
                            print("Using provider: ", provider_name)
                            chat_interface.initialize_system(provider_name)
                        except Exception as e:
                            print(f"Error: {e}")
                            raise gr.exceptions.Error(e)
                            return (f"Error: {e}",)
                        return (
                            f"Model {model_name} inference started using provider {provider_name}, and  {folder_path} loaded to DB. You can now go to Chat tab and start chatting!",
                        )

                    setup_button.click(
                        setup_chat_interface,
                        inputs=[
                            folder_path_input,
                            model_name_input,
                            provider_name,
                            api_key,
                        ],
                        outputs=setup_output,
                    )
        else:
            # Use Docker to run the chat interface, only support Ollama, no need to setup.
            os.environ["INFERENCE_MODEL"] = MODEL_NAME
            chat_interface.initialize_system("ollama")
        with gr.Tab("Chat", visible=True) as chat_tab:
            main_interface.render()
    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    freeze_support()
    interface = create_gradio_interface()
    interface.launch(
        server_name="localhost", server_port=GRADIO_SERVER_PORT, debug=True
    )
