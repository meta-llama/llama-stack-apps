# AgentStore

The AgentStore app is an example of how to create multiple agents using the llama-stack-client sdk. 
The app shows 2 agents -- one with memory ( aka RAG) and one with web search capabilities. 
We also provide functionality like chat sessions and attachment uploads. 
There is also an option to take a conversation and add it to a "Live" bank for leveraging it in future conversations.

## How to start app
1. Start your favorite llama stack distro ie. `llama stack run ...`
2. Run the app script 
```
PYTHONPATH=. python examples/agent_store/app.py --help
```
You should see some output like this
```
NAME
    app.py

SYNOPSIS
    app.py <flags>

FLAGS
    -h, --host=HOST
        Type: str
        Default: 'localhost'
    -p, --port=PORT
        Type: int
        Default: 5000
    -m, --model=MODEL
        Type: str
        Default: 'Meta-Llama3.1-8B-Instruct'
    -b, --bank_ids=BANK_IDS
        Type: str
        Default: ''
``` 
The host/port refers where your llama stack server is running.
The Memory agent can also be started by providing a list of comma separated bank-ids. 
Here is an eg .
```
PYTHONPATH=. python examples/agent_store/app.py localhost 5000

# ipv6
PYTHONPATH=. python examples/agent_store/app.py [::] 5000

# start with an existing bank-id
PYTHONPATH=. python examples/agent_store/app.py localhost 5000 --bank-ids ef9226ff-c27c-45ef-923b-f5e96b35c747
```

## How to create a Bank ? 
We provide a simple utility to create a bank.
```
PYTHONPATH=. python examples/agent_store/build_index.py --help 
```
will show 
```
NAME
    build_index.py

SYNOPSIS
    build_index.py HOST PORT FILE_DIR

POSITIONAL ARGUMENTS
    HOST
        Type: str
    PORT
        Type: int
    FILE_DIR
        Type: str

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```
##### Example 
```
PYTHONPATH=. python examples/agent_store/build_index.py localhost 5000 ~/resources/

# Sample output
PYTHONPATH=. python examples/agent_store/build_index.py localhost 5000 ~/resources/
Created bank: {
    "bank_id": "ef9226ff-c27c-45ef-923b-f5e96b35c747",
    "name": "memory_bank",
    "config": {
        "type": "vector",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size_in_tokens": 512,
        "overlap_size_in_tokens": 64
    },
    "url": null
}
Successfully created bank: ef9226ff-c27c-45ef-923b-f5e96b35c747
```
You can then start the app with this pre-filled bank(s) using 
```
PYTHONPATH=. python examples/agent_store/app.py localhost 5000 --bank-ids ef9226ff-c27c-45ef-923b-f5e96b35c747
```
