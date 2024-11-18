# RAG Evals

## 1. Run LlamaStack RAG Generation

We will generate responses from the LlamaStack RAG agent and save them to a file for evaluation. 

Edit the [.config.py](./config.py) file to set the retrieval and generation parameters.

```python
MEMORY_BANK_ID = "rag_agent_docs"

MEMORY_BANK_PARAMS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size_in_tokens": 512,
    "overlap_size_in_tokens": 64,
}

AGENT_CONFIG = AgentConfig(
    model="Llama3.1-405B-Instruct",
    instructions="You are a helpful assistant",
    sampling_params={
        "strategy": "greedy",
        "temperature": 1.0,
        "top_p": 0.9,
    },
    tools=[
        {
            "type": "memory",
            "memory_bank_configs": [{"bank_id": MEMORY_BANK_ID, "type": "vector"}],
            "query_generator_config": {"type": "default", "sep": " "},
            "max_tokens_in_context": 4096,
            "max_chunks": 10,
        }
    ],
    tool_choice="auto",
    tool_prompt_format="json",
    input_shields=[],
    output_shields=[],
    enable_session_persistence=False,
)
```

Run the generation script. This will build the memory bank index from. 

```bash
python -m evals.rag.generate localhost 5000 --docs-dir <path-to-docs-dir> --input-file-path <path-to-input-queries>
```

- `docs-dir`: Directory containing the pdfs to index
- `input-file-path`: Path to the input file containing the prompts to generate answers for

## 2. Run Scoring on LlamaStack Generated Responses

After generating responses, we will score them using llama-stack-client. This will allow you to use any of the avaialble scoring functions for running evaluations on generated response. 

```bash
llama-stack-client eval run_scoring braintrust::answer-correctness \
--dataset-path <path-to-local-dataset> \
--output-dir ./
```