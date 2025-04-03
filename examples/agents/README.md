# Llama Stack Agents Examples

This directory contains example scripts demonstrating different types of agents using the Llama Stack framework.

## Setup Instructions

1. Set API keys for external services:

This documentaion shows how to work with fireworks.ai as the inference provider. Get your fireworks API key for model inference from [here](https://fireworks.ai/account/api-keys)
   ```bash
   # For enabling agents with web search capabilities
   # Get Tavily API keys -- https://docs.tavily.com/documentation/quickstart
   export TAVILY_SEARCH_API_KEY=your_key_here
   ```
2. Start the Llama Stack server locally
   ```bash
   yes | conda create -n agents python=3.10
   conda activate agents
   pip install -U llama_stack
   
   # this will install all the dependencies to start a local llama stack server
   # pointing to fireworks for model inference
   llama stack build --template fireworks
   
   # this will start a llama stack server on localhost:8321
   llama stack run fireworks 
   ```
3. Install required Python dependencies:
```bash
pip install -r requirements.txt
```


## Available Examples

### Simple Chat Bot (`simple_chat.py`)

A basic chatbot with web search capabilities. Shows how to create a simple agent with built-in tools.

```bash
python -m examples.agents.simple_chat --host localhost --port 8321 --model_id meta-llama/Llama-3.3-70B-Instruct
```
### Multimodal Chat (`chat_multimodal.py`)

Demonstrates how to create an agent with multimodal capabilities

```bash
python -m examples.agents.chat_multimodal --host localhost --port 8321 --model_id meta-llama/Llama-3.3-70B-Instruct
```

### Chat with Documents (`chat_with_documents.py`)

Demonstrates how to create an agent that can reference and retrieve information from attached documents.

```bash
python -m examples.agents.chat_with_documents --host localhost --port 8321 --model_id meta-llama/Llama-3.3-70B-Instruct
```

### Custom Tools Integration (`agent_with_tools.py`)

Shows how to integrate custom tools with your agent, such as a calculator, stock ticker data, and custom search capabilities.

```bash
python -m examples.agents.agent_with_tools --host localhost --port 8321 --model_id meta-llama/Llama-3.3-70B-Instruct
```

### RAG Agent (`rag_agent.py`)

Demonstrates Retrieval-Augmented Generation (RAG) using vector databases for efficient information retrieval from document collections.

```bash
python -m examples.agents.rag_agent --host localhost --port 8321 --model_id meta-llama/Llama-3.3-70B-Instruct
```

### ReACT Agent (`react_agent.py`)

Implements a ReACT (Reasoning and Acting) agent that can perform multi-step reasoning and take actions based on those reasoning steps.

```bash
python -m examples.agents.react_agent --host localhost --port 8321 --model_id meta-llama/Llama-3.3-70B-Instruct
```

## Usage Tips

- All scripts accept `--host` and `--port` parameters to specify the Llama Stack server connection.
- You can specify a particular model using the `--model_id` parameter (as shown in the examples above).
- If no model is specified, the scripts will automatically select an available model.
- Look at `simple_chat` for an example of how to automatically pick an available safety shield for the agent 

For more information on the Llama Stack framework, refer to the [official documentation](https://github.com/meta-llama/llama-stack).

