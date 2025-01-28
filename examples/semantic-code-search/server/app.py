import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Set this before other imports

from flask import Flask, request, jsonify
from flask_cors import CORS
import git
from pathlib import Path
import shutil
from code_parser import CodeParser
import logging
import time

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_STORAGE = Path(__file__).parent / 'repositories'
REPO_STORAGE.mkdir(exist_ok=True)

# Initialize code parser
code_parser = CodeParser()

def clone_or_copy_repository(repo_path):
    """Clone GitHub repository or copy local directory."""
    try:
        # Create repositories directory if it doesn't exist
        repo_dir = Path(__file__).parent / "repositories"
        repo_dir.mkdir(exist_ok=True)
        
        # Clean up repo_path to handle GitHub URLs
        if repo_path.startswith(('http://', 'https://')):
            # Remove /tree/branch-name if present
            if '/tree/' in repo_path:
                repo_path = repo_path.split('/tree/')[0]
            # Extract repository name from URL
            repo_name = repo_path.rstrip('/').split('/')[-1]
            target_dir = repo_dir / repo_name
        else:
            # For local paths, use the last directory name
            repo_name = Path(repo_path).name
            target_dir = repo_dir / repo_name
            
        # Remove target directory if it exists
        if target_dir.exists():
            shutil.rmtree(target_dir)
            
        # Clone repository or copy local directory
        if repo_path.startswith(('http://', 'https://')):
            git.Repo.clone_from(repo_path, target_dir)
        else:
            shutil.copytree(repo_path, target_dir)
            
        return target_dir
        
    except Exception as e:
        logger.error(f"Error in clone_or_copy_repository: {str(e)}")
        raise

@app.route('/api/index', methods=['POST'])
def index_repository():
    data = request.json
    repo_path = data.get('repository')
    
    if not repo_path:
        return jsonify({'error': 'Repository path is required'}), 400
    
    try:
        # Set a longer timeout for the request
        request.environ.get('werkzeug.server.shutdown')
        
        # Clone/copy the repository
        local_path = clone_or_copy_repository(repo_path)
        
        # Index the repository using Tree-sitter
        print(f"Starting indexing of repository: {repo_path}")
        indexed_files = code_parser.index_directory(local_path)
        print(f"Finished indexing repository: {repo_path}")
        
        return jsonify({
            'status': 'success',
            'message': 'Repository indexed successfully',
            'indexed_files': indexed_files
        })
        
    except Exception as e:
        error_msg = f'Failed to index repository: {str(e)}'
        print(f"Error: {error_msg}")
        return jsonify({
            'error': error_msg
        }), 500

@app.route('/api/search', methods=['POST'])
def search_code():
    logger.info("Received search request")
    data = request.json
    query = data.get('query', '')
    
    logger.debug(f"Search query: {query[:100]}...")  # Log first 100 chars of query
    
    if not query:
        logger.warning("Empty query received")
        return jsonify({'error': 'Query is required'}), 400
        
    try:
        # Get query embedding and find similar code snippets
        logger.debug("Starting similarity search")
        start_time = time.time()
        
        results = code_parser.embeddings.search_similar_code(query, n_results=1)
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f} seconds")
        
        if results:
            logger.debug(f"Number of results: {len(results)}")
            logger.debug(f"Top result similarity score: {results[0]['similarity']:.4f}")
            
            # Format response
            formatted_results = []
            for result in results:
                full_path = result['metadata'].get('file_path', '')
                # Extract the path after 'repositories/'
                relative_path = full_path.split('repositories/')[-1] if 'repositories/' in full_path else full_path
                
                formatted_results.append({
                    'code': result['code'],
                    'file': relative_path,
                    'similarity': f"{result['similarity']*100:.1f}%",
                    'docstring': result['metadata'].get('docstring', '')
                })
            
            return jsonify({
                'results': formatted_results,
                'search_time': search_time
            })
        else:
            logger.warning("No results found")
            return jsonify({
                'results': [],
                'search_time': search_time
            })
            
    except Exception as e:
        error_msg = f'Error during search: {str(e)}'
        logger.error(error_msg)
        logger.error(f"Stack trace:", exc_info=True)
        return jsonify({'error': error_msg}), 500

@app.route('/api/verify', methods=['GET'])
def verify_embeddings():
    """Verify the integrity of stored embeddings."""
    try:
        result = code_parser.embeddings.verify_embeddings()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Enable debug mode and auto-reloader for development
    app.config['DEBUG'] = False
    app.config['ENV'] = 'development'
    app.run(host='0.0.0.0', port=5000)
