# RAG System Evaluation

This directory contains tools for evaluating the Retrieval-Augmented Generation (RAG) system using RAGAS metrics.

## Setup

1. Create your environment file:

```bash
cp .env.template .env
```

2. Configure the environment variables in `.env`:

```env
# Server Configuration
HOST=localhost          # Your server host
PORT=5000              # Your server port
CHROMA_PORT=8000       # Chroma DB port

# Model and Memory Configuration
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct  # Model to use
MEMORY_BANK_ID=eval_bank                      # Memory bank identifier

# File Paths
DOCS_DIR=../output     # Directory containing your documents
```

## Running the Evaluation

1. Make sure your server is running and accessible at the configured host and port.

2. Run the evaluation script:

```bash
python eval.py
```

The script will:

- Set up a memory bank for evaluation
- Load your documents
- Generate responses for test questions
- Evaluate the responses using various RAGAS metrics:
  - Context Precision
  - Context Recall
  - Faithfulness
  - Answer Relevancy
  - Factual Correctness
  - Semantic Similarity

Results will be saved to `evaluation_results.csv`.

## Analyzing Results

For detailed analysis of your evaluation results, you can use the Jupyter notebook:

```bash
jupyter notebook explain-eval.ipynb
```

The notebook provides:

- Visualization of evaluation metrics
- Detailed breakdown of each metric
- Analysis of system performance
- Insights for improvement

## Metrics Explanation

The evaluation uses the following RAGAS metrics:

1. **Context Precision**: Measures how much of the retrieved context is actually relevant
2. **Context Recall**: Measures if all relevant information was retrieved
3. **Faithfulness**: Measures if the answer is faithful to the provided context
4. **Answer Relevancy**: Measures if the answer is relevant to the question
5. **Factual Correctness**: Measures the factual accuracy of the answer
6. **Semantic Similarity**: Measures semantic closeness between answer and reference

## Troubleshooting

If you encounter issues:

1. Verify your server is running and accessible
2. Check the environment variables in `.env`
3. Ensure your documents are in the correct directory
4. Check the logs for detailed error messages

## Requirements

- Python 3.10+
- Jupyter Notebook (for analysis)
- Required Python packages (install via `pip`):
  - ragas
  - datasets
  - pandas
  - numpy
  - matplotlib (for visualization)
