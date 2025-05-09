# BookMind

[![thumbnail](https://github.com/user-attachments/assets/11c6f1f3-59db-4638-9b1a-68f1d25efec4)](https://youtu.be/DL4-DswxfEM)

BookMind is a web application that allows users to explore character relationships and storylines in books using AI-powered visualizations. The application provides interactive mind maps, AI chatbots for deep questions, book summaries, and community contributions.

## Features

- Interactive Mind Maps: Visualize relationships between characters and plot elements.
- AI Chatbot: Ask deep questions about the book and get insightful answers.
- Book Summaries: Get concise overviews of plots and themes.
- Community Contributions: Add and refine maps with fellow book lovers.

## Prerequisites

- Node.js
- Python >= 3.10
- LlamaStack server running locally
- Environment variables:
  - LLAMA_STACK_PORT
  - INFERENCE_MODEL
  - REACT_APP_GOOGLE_BOOKS_API_KEY

## Getting Started

### Run llama-stack

1. Setting up Ollama server  
   Please check the [Ollama Documentation](https://github.com/ollama/ollama) on how to install and run Ollama. After installing Ollama, you need to run `ollama serve` to start the server.

```
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
```

2. Running `llama-stack` server

```
pip install llama-stack

export LLAMA_STACK_PORT=5000

# This builds llamastack-ollama conda environment
llama stack build --template ollama --image-type conda

conda activate llamastack-ollama

llama stack run \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434 \
  ollama
```

### Backend Setup

1. Install dependencies:

```
cd server
pip install -r requirements.txt
```

2. Set `.env` in the `server` directory

You should modify the name of `example.env` to `.env` in the `server` directory.  
**Modify INFERENCE_MODEL in example.env with yours.**

3. Run the server:

```
python server.py
```

### Frontend Setup

1. Set up GOOGLE_BOOKS_API_KEY:

You should rename `example.env` with `.env` and replace `{YOUR_API_KEY}` with your [google_books_api_key](https://developers.google.com/books/docs/v1/using) after getting your api key.

```
REACT_APP_GOOGLE_BOOKS_API_KEY={YOUR_API_KEY}
```

2. Install dependencies and run the application:

```
npm install
npm start
```

## Usage

1. Initialize Memory: Upload your book or choose from the library to initialize memory.
2. AI Analysis: The AI analyzes the book and generates a mind map.
3. Explore Insights: Explore relationships, themes, and Q&A insights.

## What did we use Llama-stack in BookMind?

1️⃣ Llama Inference models: You can start a LLM application using various LLM services easily.  
2️⃣ RAG with FAISS: We leveraged FAISS in Llama-stack for Retrieval-Augmented Generation, enabling real-time responses to character relationship questions.  
3️⃣ Multi-Hop Reasoning: Our system performs sequential inference—first extracting characters and relationships, then generating graphized mind map data in JSON for visual storytelling.

## Contributors

[Original Repo](https://github.com/seyeong-han/BookMind)  
[seyeong-han](https://github.com/seyeong-han)  
[sunjinj](https://github.com/SunjinJ)  
[WonHaLee](https://github.com/WonHaLee)
