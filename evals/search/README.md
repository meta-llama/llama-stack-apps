# Search Agent Evaluation

## 1. Bulk Generation for Search Agent

Edit the `config.py` file to change the model and other parameters.
```python
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
            "type": "brave_search",
            "engine": "brave",
            "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
        }
    ],
    tool_choice="auto",
    tool_prompt_format="json",
    input_shields=[],
    output_shields=[],
    enable_session_persistence=False,
)
```

Run the generation script.

```bash
python -m evals.search.generate localhost 5000 <path-to-input-queries>
```

Example input dataset: 
- [llamastack/evals/evals__simpleqa](https://huggingface.co/datasets/llamastack/evals/viewer/evals__simpleqa)


## 2. Scoring with LlamaStack Client

```bash
llama-stack-client eval run_scoring llm-as-judge::llm-as-judge-simpleqa \
--dataset-path <path-to-local-dataset> \
--output-dir ./
```