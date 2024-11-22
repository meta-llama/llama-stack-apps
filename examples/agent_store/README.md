# AgentStore

The AgentStore app is an example of how to create multiple agents using the llama-stack-client sdk.
The app shows 2 agents -- one with memory ( aka RAG) and one with web search capabilities.
We also provide functionality like chat sessions and attachment uploads.
There is also an option to take a conversation and add it to a "Live" bank for leveraging it in future conversations.

## Installtion
1. Git clone this repo
```
git clone https://github.com/meta-llama/llama-stack-apps.git
```
2. Start new env
```
conda create -n agentstore python=3.10
conda activate agentstore
cd <path/to/llama-stack-apps>
pip install -r requirements_agent_store.txt
```

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
$ llama-stack-client memory_banks list

+--------------+---------------+--------+-------------------+------------------------+--------------------------+
| identifier   | provider_id   | type   | embedding_model   |   chunk_size_in_tokens |   overlap_size_in_tokens |
+==============+===============+========+===================+========================+==========================+
| memory_bank  | meta0         | vector | all-MiniLM-L6-v2  |                    512 |                       64 |
+--------------+---------------+--------+-------------------+------------------------+--------------------------+

$ PYTHONPATH=. python examples/agent_store/app.py localhost 5000 --bank-ids memory_bank
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

Successfully created bank: memory_bank
```
You can then start the app with this pre-filled bank(s) using
```
PYTHONPATH=. python examples/agent_store/app.py localhost 5000 --bank-ids memory_bank
```

The bank-ids can be obtained by running
```
$ llama-stack-client memory_banks list

+--------------+---------------+--------+-------------------+------------------------+--------------------------+
| identifier   | provider_id   | type   | embedding_model   |   chunk_size_in_tokens |   overlap_size_in_tokens |
+==============+===============+========+===================+========================+==========================+
| memory_bank  | meta0         | vector | all-MiniLM-L6-v2  |                    512 |                       64 |
+--------------+---------------+--------+-------------------+------------------------+--------------------------+
```
