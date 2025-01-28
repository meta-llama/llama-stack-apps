# Semantic Code Search

A powerful desktop application that enables semantic code search across your repositories using state-of-the-art language models and embeddings.

## Features

- **Repository Integration**: Support for both GitHub repositories and local codebases
- **Semantic Search**: Advanced code search using natural language queries
- **Real-time Indexing**: Fast and efficient code indexing with embeddings
- **Cross-platform**: Built with Electron for Windows, macOS, and Linux support
- **Modern UI**: Clean and intuitive interface built with React

## Tech Stack

- **Backend**: Flask Python server with embeddings-based search
- **Frontend**: Electron + React application
- **Search**: Semantic code search powered by language models
- **Storage**: Local embeddings storage for quick searches

## Project Structure

```
semantic-code-search/
├── frontend/           # Electron + React frontend application
│   ├── src/           # React source code
│   ├── preload/       # Electron preload scripts
│   ├── resources/     # Application resources (icons, etc.)
│   ├── build/         # Build output directory
│   ├── out/          # Distribution output directory
│   └── package.json   # Frontend dependencies
├── server/            # Flask backend server
│   ├── app.py         # Main server application
│   ├── embeddings.py  # Code embedding generation and search
│   └── code_parser.py # Code parsing and processing
├── README.md         # Project documentation
└── environment.yml   # Conda environment for backend
```

## Getting Started

### Prerequisites

- Conda (for Python environment management)
- Node.js 16+
- Git

### Installation

1. Clone the repository

2. Set up the backend:
   ```bash
   # Create and activate the Conda environment from environment.yml
   conda env create -f environment.yml
   conda activate semantic-code-search
   ```

3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

1. Start the backend server:
   ```bash
   conda activate semantic-code-search
   cd server
   python app.py
   ```

2. Start the frontend application (in a new terminal):
   ```bash
   cd frontend
   npm run dev
   ```

## Usage

1. Launch the application
2. Add a repository (GitHub URL or local path)
3. Wait for the indexing to complete
4. Use natural language to search through your codebase

## Development

### Backend (`/server`)
- Flask Python server with embeddings-based search
- Code parsing and embedding generation
- Repository management and indexing

### Frontend (`/frontend`)
- Electron + React application
- Modern UI components
- Real-time search interface
- Repository management UI

### Managing Dependencies

To update the environment.yml file after installing new packages:
```bash
conda env export --name semantic-code-search > environment.yml
```

To recreate the environment on another machine:
```bash
conda env create -f environment.yml
```

## License

MIT License
