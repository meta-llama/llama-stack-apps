from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import git
from pathlib import Path
import shutil
from code_parser import CodeParser

app = Flask(__name__)
CORS(app)

REPO_STORAGE = Path(__file__).parent / 'repositories'
REPO_STORAGE.mkdir(exist_ok=True)

# Initialize code parser
code_parser = CodeParser()

def clone_or_copy_repository(repo_path):
    """Clone GitHub repository or copy local directory."""
    if repo_path.startswith('https://github.com'):
        repo_name = repo_path.split('/')[-1]
        target_path = REPO_STORAGE / repo_name
        if target_path.exists():
            shutil.rmtree(target_path)
        git.Repo.clone_from(repo_path, target_path)
        return str(target_path)
    else:
        # For local paths, we'll create a symlink instead of copying
        repo_name = Path(repo_path).name
        target_path = REPO_STORAGE / repo_name
        if target_path.exists():
            if target_path.is_symlink():
                target_path.unlink()
            else:
                shutil.rmtree(target_path)
        target_path.symlink_to(repo_path, target_is_directory=True)
        return str(target_path)

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
    data = request.json
    query = data.get('query', '')
    # TODO: Implement actual semantic search logic
    return jsonify({
        'results': [
            {
                'file': 'example.py',
                'snippet': 'def example_function():',
                'score': 0.95
            }
        ]
    })

@app.route('/api/verify', methods=['GET'])
def verify_embeddings():
    """Verify the integrity of stored embeddings."""
    print("Verify endpoint hit!")  
    try:
        print("Getting statistics...")  
        # Get statistics about stored embeddings
        stats = code_parser.embeddings.verify_embeddings()
        
        print("Getting samples...")  
        # Get sample items for manual verification
        samples = code_parser.embeddings.get_sample_items(n_samples=3)
        
        print("Returning response...")  
        return jsonify({
            'status': 'success',
            'statistics': stats,
            'samples': samples
        })
        
    except Exception as e:
        print(f"Error in verify endpoint: {str(e)}")  
        return jsonify({
            'error': f'Failed to verify embeddings: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Disable debug mode and auto-reloader
    app.config['DEBUG'] = False
    app.config['ENV'] = 'production'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
