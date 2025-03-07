# Book Character Graph API

A Flask-based API server that analyzes books to create character relationship graphs and provides an interactive query interface using LlamaStack.

## Features

- Character and relationship extraction from books
- Graph generation of character relationships
- Memory-based query system for book details
- Interactive Q&A about book characters and plots

## Prerequisites

- Python 3.x
- LlamaStack server running locally
- Environment variables:
  - `LLAMA_STACK_PORT`
  - `INFERENCE_MODEL`

## Get Started

```bash
# Install dependencies
pip install -r requirements.txt

python server.py
```
