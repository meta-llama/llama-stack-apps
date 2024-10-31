# Running Examples

This folder contains some example client scripts using our Python SDK for client to connect with Llama Stack Distros. To run the example scripts:

## Step 0. Start Server
- Follow steps in our [Getting Started](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.md) guide to setup a Llama Stack server.

## Step 1. Run Client
First, setup depenencies via
```
conda create -n app python=3.10
cd <path-to-llama-stack-apps-repo>
conda activate app

# Install dependencies
pip install -r requirements.txt
```

Run client script via connecting to your Llama Stack server
```
python -m examples.agents.hello localhost 5000
```

## Demo Scripts
```
python -m examples.agents.hello localhost 5000
python -m examples.agents.inflation localhost 5000
python -m examples.agents.llama_guard_demo localhost 5000
python -m examples.agents.podcast_transcript localhost 5000
python -m examples.agents.rag_as_attachments localhost 5000
python -m examples.agents.rag_with_memory_bank localhost 5000
python -m examples.agents.vacation localhost 5000
```

### Simple API Scripts
```
python examples/inference/client.py localhost 5000
python examples/memory/client.py localhost 5000
python examples/safety/client.py localhost 5000
```

# Demo Apps
```
PYTHONPATH=. with-proxy mesop app/main.py
```

```
PYTHONPATH=. with-proxy gradio examples/agent_store/app.py
```
