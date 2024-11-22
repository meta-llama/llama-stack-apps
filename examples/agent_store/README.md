# AgentStore

The AgentStore app is an example of how to create multiple agents using the llama-stack-client sdk.
The app shows 2 agents -- one with memory ( aka RAG) and one with web search capabilities.
We also provide functionality like chat sessions and attachment uploads.
There is also an option to take a conversation and add it to a "Live" bank for leveraging it in future conversations.

## Installation
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
python -m examples.agent_store.app --help
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
        Default: 'meta-llama/Llama-3.1-8B-Instruct'
    -b, --bank_ids=BANK_IDS
        Type: str
        Default: ''
```
The host/port refers where your llama stack server is running.
The Memory agent can also be started by providing a list of comma separated bank-ids.

To start the app without any pre-existing bank-ids
```
$ python -m examples.agent_store.app localhost 5000

...
* Running on local URL:  http://0.0.0.0:7860
```

## How to create a memory bank ?
We provide a simple utility to create a bank.
```
python -m examples.agent_store.build_index --help
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
python -m examples.agent_store.build_index localhost 5000 ~/resources/

# Sample output
python -m examples.agent_store.build_index localhost 5000 ~/resources/

Successfully created bank: memory_bank
```
You can then start the app with this pre-filled bank(s) using
```
python -m examples.agent_store.app localhost 5000 --bank-ids memory_bank
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

### How to run evaluation ?
First, you will need to provide a dataset of user input queries to evaluate on. In the `bulk_generate` script, the script will generate responses using the Memory agent from the app using supplied offline docs and user input query prompts to generate responses.

Running the above script will generate a new dataset with the generated responses and save it for scoring.

```
python -m examples.agent_store.eval.bulk_generate --docs-dir <path/to/docs> --dataset-path <path/to/dataset>
```
- `--docs-dir` is the directory containing the documents to build the memory bank on for retrieval.
- `--dataset-path` is the path to the dataset of user input queries to evaluate on.

You will see example outputs:

```
$ python -m examples.agent_store.eval.bulk_generate --docs-dir <path/to/docs> --dataset-path <path/to/dataset>
...
You may now run `llama-stack-client eval run_scoring <scoring_fn_ids> --dataset_path <path/to/dataset>` to score the generated responses.
```

Using the generated dataset, you can now score the responses using the `llama-stack-client` CLI for evaluating generated responses.

```
llama-stack-client eval run_scoring <scoring_fn_ids> --dataset_path <path/to/dataset>
```
