# Python SDK

This folder contains some example client scripts using our Python SDK for client to connect with Llama Stack Distros. To run the example scripts:

# Step 0. Start Server
- Follow steps in our [Getting Started]() guide to setup a Llama Stack server.

# Step 1. Run Client
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
python -m sdk_examples.agentic_system.client localhost 5000
```

You should be able to see stdout of the form ---


https://github.com/user-attachments/assets/113fa3ee-d8cb-4dc3-ae0a-8a8064f7a622

