# Evals API Example

Example of using Llama Stack evals API to run evals on eval dataset for RAG agent. The demo includes:

1. Bulk generation using agent with memory for RAG
2. Using a custom evals dataset to run generation and evaluate with LLM as Judge score mertrics.

## Run
#### (Optional) Run RAG Agent Bulk Generation
- Run generation with RAG agent app for preparing evals dataset.
```
python -m examples.evals.rag_agent.main <host> <port> \
--file-dir <file_dir>
--app-dataset-path <app_dataset_path>
```
- `file_dir` is directory containing *.pdf files for
- `app_dataset_path` is the app evaluation dataset file containing input_query for generation and the expected_answer

This will prepare a evaluation dataset to be used for metrics computation with the following columns saved to `rag_evals.xlsx`.
- `generated_answer`
- `expected_answer`
- `input_query`

Next, we'll register the dataset to a Llama Stack server for computing eval metrics.

#### Run Evals Scoring
```
python -m examples.evals.rag_agent.main <host> <port> \
--eval-dataset-path <eval_dataset_path>
```
- `eval_dataset_path` is the eval dataset with the columns `generated_answer`, `expected_answer`, `input_query` to be used for metrics computation.

**Example Output**
```
datasets/create: DatasetCreateResponse(status='fail', msg='Dataset rag-evals already exists.')
LlamaStackLLMJudgeScorer:avg_judge_score: 3.1666666666666665
BraintrustAnswerCorrectnessScorer:avg_correctness_score: 0.565270561846977
```
