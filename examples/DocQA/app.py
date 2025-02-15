import os
import subprocess
import threading
import time
from multiprocessing import freeze_support

import customtkinter as ctk
from dotenv import load_dotenv

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Document
from llama_stack_client.types.agent_create_params import AgentConfig

# Set CustomTkinter to light mode to mimic OpenAI's website
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")  # Adjust color theme as needed

# Load environment variables and set defaults
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
DOCS_DIR = "/root/rag_data/output" if os.getenv("USE_GPU_FOR_DOC_INGESTION", False) else "/root/rag_data/"

class LlamaChatInterface:
    def __init__(self):
        self.docs_dir = None
        self.client = None
        self.agent = None
        self.session_id = None
        self.vector_db_id = "docqa_vector_db"

    def initialize_system(self, provider_name="ollama"):
        self.client = LlamaStackAsLibraryClient(provider_name)
        # Remove scoring and eval providers.
        del self.client.async_client.config.providers["scoring"]
        del self.client.async_client.config.providers["eval"]
        tool_groups = []
        for provider in self.client.async_client.config.tool_groups:
            if provider.provider_id == "rag-runtime":
                tool_groups.append(provider)
        self.client.async_client.config.tool_groups = tool_groups
        self.client.initialize()
        self.docs_dir = DOCS_DIR
        self.setup_vector_dbs()
        self.initialize_agent()

    def setup_vector_dbs(self):
        providers = self.client.providers.list()
        vector_io_provider = [provider for provider in providers if provider.api == "vector_io"]
        provider_id = vector_io_provider[0].provider_id
        vector_dbs = self.client.vector_dbs.list()
        if vector_dbs and any(bank.identifier == self.vector_db_id for bank in vector_dbs):
            print(f"vector_dbs '{self.vector_db_id}' exists.")
        else:
            print(f"vector_dbs '{self.vector_db_id}' does not exist. Creating...")
            self.client.vector_dbs.register(
                vector_db_id=self.vector_db_id,
                embedding_model="all-MiniLM-L6-v2",
                embedding_dimension=384,
            )
            self.load_documents()
            print("vector_dbs registered.")

    def load_documents(self):
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
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=256,
            )
            print(f"Loaded {len(documents)} documents from {self.docs_dir}")

    def initialize_agent(self):
        agent_config = AgentConfig(
            model=MODEL_NAME,
            instructions="You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
            toolgroups=[{
                "name": "builtin::rag",
                "args": {"vector_db_ids": [self.vector_db_id]},
            }],
            enable_session_persistence=True,
        )
        self.agent = Agent(self.client, agent_config)
        self.session_id = self.agent.create_session("session-docqa")

    def chat_stream(self, message: str, history):
        try:
            history = history or []
            history.append([message, ""])
            response = self.agent.create_turn(
                messages=[{"role": "user", "content": message}],
                session_id=self.session_id,
            )
        except Exception as e:
            print(f"Error: {e}")
            yield f"Error: {e}"
            return
        current_response = ""
        for log in EventLogger().log(response):
            if hasattr(log, "content"):
                if "tool_execution>" not in str(log):
                    current_response += log.content
                    history[-1][1] = current_response
                    yield history.copy()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("LlamaStack Chat")
        self.geometry("1000x700")
        self.configure(padx=20, pady=20)
        
        self.chat_interface = LlamaChatInterface()
        self.chat_history = []  # List of [user_message, assistant_message]
        self.setup_completed = False

        # Header Frame
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(pady=(10, 20), fill="x")
        self.header_label = ctk.CTkLabel(
            self.header_frame,
            text="LlamaStack Chat",
            font=("Inter", 28, "bold")
        )
        self.header_label.pack()

        # Tabview for Setup and Chat
        self.tabview = ctk.CTkTabview(self, width=960, height=580, corner_radius=10)
        self.tabview.pack(pady=10)
        self.tabview.add("Setup")
        self.tabview.add("Chat")

        # ------------------- Setup Tab -------------------
        self.setup_tab = self.tabview.tab("Setup")
        self.setup_inner_frame = ctk.CTkFrame(self.setup_tab, corner_radius=10)
        self.setup_inner_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.setup_folder_label = ctk.CTkLabel(self.setup_inner_frame, text="Data Folder Path:", font=("Inter", 16))
        self.setup_folder_label.pack(pady=8)
        self.folder_entry = ctk.CTkEntry(self.setup_inner_frame, width=500, font=("Inter", 14))
        self.folder_entry.pack(pady=8)
        self.folder_entry.insert(0, DOCS_DIR)

        self.model_label = ctk.CTkLabel(self.setup_inner_frame, text="Llama Model Name:", font=("Inter", 16))
        self.model_label.pack(pady=8)
        self.model_combobox = ctk.CTkComboBox(
            self.setup_inner_frame,
            width=400,
            font=("Inter", 14),
            values=[
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-70B-Instruct",
                "meta-llama/Llama-3.1-405B-Instruct",
            ]
        )
        self.model_combobox.pack(pady=8)
        self.model_combobox.set("meta-llama/Llama-3.2-1B-Instruct")

        self.provider_label = ctk.CTkLabel(self.setup_inner_frame, text="Provider List:", font=("Inter", 16))
        self.provider_label.pack(pady=8)
        self.provider_combobox = ctk.CTkComboBox(
            self.setup_inner_frame,
            width=400,
            font=("Inter", 14),
            values=[
                "ollama",
                "together",
                "fireworks",
            ]
        )
        self.provider_combobox.pack(pady=8)
        self.provider_combobox.set("ollama")

        self.api_label = ctk.CTkLabel(self.setup_inner_frame, text="API Key (if needed):", font=("Inter", 16))
        self.api_label.pack(pady=8)
        self.api_entry = ctk.CTkEntry(self.setup_inner_frame, width=500, font=("Inter", 14), show="*")
        self.api_entry.pack(pady=8)

        self.setup_button = ctk.CTkButton(
            self.setup_inner_frame,
            text="Setup Chat Interface",
            font=("Inter", 16, "bold"),
            command=self.setup_chat_interface,
            corner_radius=8
        )
        self.setup_button.pack(pady=20)

        self.setup_status_label = ctk.CTkLabel(self.setup_inner_frame, text="", font=("Inter", 14))
        self.setup_status_label.pack(pady=8)

        # ------------------- Chat Tab -------------------
        self.chat_tab = self.tabview.tab("Chat")
        self.chat_inner_frame = ctk.CTkFrame(self.chat_tab, corner_radius=10)
        self.chat_inner_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Chat display using CTkTextbox; we will access its internal _textbox for tags.
        self.chat_display = ctk.CTkTextbox(
            self.chat_inner_frame,
            width=920,
            height=400,
            font=("Inter", 14),
            fg_color="white",      # Light background
            text_color="black"
        )
        self.chat_display.pack(pady=10)
        # Configure tags on the underlying tkinter.Text widget:
        self.chat_display._textbox.tag_configure("user", foreground="#0066FF", font=("Inter", 14, "bold"))
        self.chat_display._textbox.tag_configure("assistant", foreground="#008800", font=("Inter", 14))
        self.chat_display.configure(state="disabled")

        self.message_entry = ctk.CTkEntry(self.chat_inner_frame, width=700, font=("Inter", 14))
        self.message_entry.pack(pady=8)
        self.message_entry.bind("<Return>", lambda event: self.send_message())

        self.button_frame = ctk.CTkFrame(self.chat_inner_frame)
        self.button_frame.pack(pady=10)

        self.send_button = ctk.CTkButton(
            self.button_frame,
            text="Send",
            font=("Inter", 14, "bold"),
            command=self.send_message,
            corner_radius=8
        )
        self.send_button.pack(side="left", padx=10)

        self.clear_button = ctk.CTkButton(
            self.button_frame,
            text="Clear",
            font=("Inter", 14, "bold"),
            command=self.clear_chat,
            corner_radius=8
        )
        self.clear_button.pack(side="left", padx=10)

        self.exit_button = ctk.CTkButton(
            self.button_frame,
            text="Exit",
            font=("Inter", 14, "bold"),
            command=self.destroy,
            corner_radius=8
        )
        self.exit_button.pack(side="left", padx=10)

    def setup_chat_interface(self):
        folder_path = self.folder_entry.get()
        model_name = self.model_combobox.get()
        provider_name = self.provider_combobox.get()
        api_key = self.api_entry.get()

        global MODEL_NAME, DOCS_DIR
        DOCS_DIR = folder_path
        MODEL_NAME = model_name
        os.environ["INFERENCE_MODEL"] = model_name  # Set inference model environment variable

        if not os.path.exists(folder_path):
            self.setup_status_label.configure(text=f"Folder {folder_path} does not exist.", text_color="red")
            return

        if provider_name == "ollama":
            ollama_name_dict = {
                "meta-llama/Llama-3.2-1B-Instruct": "llama3.2:1b-instruct-fp16",
                "meta-llama/Llama-3.2-3B-Instruct": "llama3.2:3b-instruct-fp16",
                "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b-instruct-fp16",
            }
            if model_name not in ollama_name_dict:
                self.setup_status_label.configure(text=f"Model {model_name} is not supported. Use 1B, 3B, or 8B.", text_color="red")
                return
            ollama_name = ollama_name_dict[model_name]
            try:
                print("Starting Ollama server...")
                subprocess.Popen(
                    f"/usr/local/bin/ollama run {ollama_name} --keepalive=99h".split(),
                    stdout=subprocess.DEVNULL,
                )
                time.sleep(3)
            except Exception as e:
                self.setup_status_label.configure(text=f"Error: {e}", text_color="red")
                return
        elif provider_name == "together":
            os.environ["TOGETHER_API_KEY"] = api_key
        elif provider_name == "fireworks":
            os.environ["FIREWORKS_API_KEY"] = api_key

        try:
            print("Initializing LlamaStack client...")
            self.chat_interface.initialize_system(provider_name)
            self.setup_status_label.configure(text=f"Model {model_name} started using provider {provider_name}.", text_color="green")
            self.setup_completed = True
        except Exception as e:
            self.setup_status_label.configure(text=f"Error during setup: {e}", text_color="red")

    def send_message(self):
        if not self.setup_completed:
            self.append_chat("System: Please complete setup first.\n")
            return

        message = self.message_entry.get().strip()
        if not message:
            return
        # Append the user message to chat history
        self.chat_history.append([message, ""])
        self.update_chat_display()
        self.message_entry.delete(0, "end")
        threading.Thread(target=self.process_chat, args=(message,), daemon=True).start()

    def process_chat(self, message):
        generator = self.chat_interface.chat_stream(message, self.chat_history)
        try:
            for history in generator:
                self.chat_history = history
                self.after(0, self.update_chat_display)
                time.sleep(0.1)
        except Exception as e:
            self.after(0, lambda error=e: self.append_chat(f"\nError: {error}\n"))

    def update_chat_display(self):
        # Enable the underlying textbox for text updates.
        self.chat_display.configure(state="normal")
        self.chat_display._textbox.delete("1.0", "end")
        for entry in self.chat_history:
            self.chat_display._textbox.insert("end", f"User: {entry[0]}\n", "user")
            self.chat_display._textbox.insert("end", f"Assistant: {entry[1]}\n\n", "assistant")
        self.chat_display.configure(state="disabled")
        self.chat_display._textbox.see("end")

    def clear_chat(self):
        self.chat_history = []
        self.update_chat_display()

    def append_chat(self, text):
        self.chat_display.configure(state="normal")
        self.chat_display._textbox.insert("end", text)
        self.chat_display.configure(state="disabled")
        self.chat_display._textbox.see("end")

if __name__ == "__main__":
    freeze_support()
    app = App()
    app.mainloop()
