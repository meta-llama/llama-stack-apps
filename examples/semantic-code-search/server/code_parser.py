from tree_sitter import Language, Parser
from pathlib import Path
import os
import subprocess
import json
import hashlib
from embeddings import CodeEmbeddings

class CodeParser:
    def __init__(self):
        self.parser = None
        self.languages = {}
        self.embeddings = CodeEmbeddings()
        self.setup_tree_sitter()

    def setup_tree_sitter(self):
        """Initialize Tree-sitter and build language parsers."""
        # Create languages directory if it doesn't exist
        languages_dir = Path(__file__).parent / "tree-sitter-langs"
        languages_dir.mkdir(exist_ok=True)
        
        # Dictionary of supported languages and their repositories
        lang_repos = {
            'python': 'https://github.com/tree-sitter/tree-sitter-python',
            'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript',
        }

        # Build each language
        for lang_name, repo_url in lang_repos.items():
            lang_dir = languages_dir / f"tree-sitter-{lang_name}"
            
            # Clone language repository if not exists
            if not lang_dir.exists():
                subprocess.run(['git', 'clone', repo_url, str(lang_dir)], check=True)

        # Build the languages library
        Language.build_library(
            str(languages_dir / "languages.so"),
            [str(languages_dir / f"tree-sitter-{lang}") for lang in lang_repos.keys()]
        )

        # Load languages
        for lang_name in lang_repos.keys():
            self.languages[lang_name] = Language(
                str(languages_dir / "languages.so"), lang_name
            )

        self.parser = Parser()

    def parse_file(self, file_path):
        """Parse a source code file and return its AST."""
        file_path = Path(file_path)
        
        # Determine language based on file extension
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript'
        }
        
        ext = file_path.suffix.lower()
        if ext not in ext_to_lang:
            return None
            
        lang_name = ext_to_lang[ext]
        self.parser.set_language(self.languages[lang_name])
        
        with open(file_path, 'rb') as f:
            source_code = f.read()
            
        tree = self.parser.parse(source_code)
        return tree

    def extract_code_elements(self, tree, source_code):
        """Extract relevant code elements from the AST."""
        code_elements = []
        
        def traverse(node, depth=0):
            if node.type in ['function_definition', 'class_definition', 'method_definition']:
                code_elements.append({
                    'type': node.type,
                    'name': source_code[node.child_by_field_name('name').start_byte:
                                     node.child_by_field_name('name').end_byte].decode('utf8'),
                    'start_point': node.start_point,
                    'end_point': node.end_point,
                    'code': source_code[node.start_byte:node.end_byte].decode('utf8')
                })
            
            for child in node.children:
                traverse(child, depth + 1)
                
        traverse(tree.root_node)
        return code_elements

    def index_file(self, file_path):
        """Parse and index a single file."""
        file_path = Path(file_path)
        if not file_path.exists():
            return None
            
        with open(file_path, 'rb') as f:
            source_code = f.read()
            
        tree = self.parse_file(file_path)
        if tree is None:
            return None
            
        code_elements = self.extract_code_elements(tree, source_code)
        
        # Store each code element with embeddings
        for element in code_elements:
            # Create a unique ID for the code snippet
            snippet_id = hashlib.sha256(
                f"{file_path}:{element['start_point']}:{element['end_point']}".encode()
            ).hexdigest()
            
            # Prepare metadata
            metadata = {
                'file_path': str(file_path),
                'type': element['type'],
                'name': element['name'],
                'start_line': element['start_point'][0],
                'end_line': element['end_point'][0],
            }
            
            # Store in vector database
            self.embeddings.store_code_snippet(
                snippet_id=snippet_id,
                code_snippet=element['code'],
                metadata=metadata
            )
            
        return code_elements

    def index_directory(self, directory_path):
        """Recursively index all supported files in a directory."""
        directory_path = Path(directory_path)
        indexed_files = {}
        
        try:
            for ext in ['.py', '.js', '.ts']:
                for file_path in directory_path.rglob(f'*{ext}'):
                    try:
                        # Skip hidden files and directories
                        if any(part.startswith('.') for part in file_path.parts):
                            continue
                            
                        print(f"Indexing file: {file_path}")  # Progress tracking
                        result = self.index_file(file_path)
                        if result is not None:
                            indexed_files[str(file_path)] = result
                            
                    except Exception as e:
                        print(f"Error indexing file {file_path}: {e}")
                        # Continue with next file instead of failing completely
                        continue
                        
            return indexed_files
            
        except Exception as e:
            print(f"Error during directory indexing: {e}")
            return indexed_files  # Return any files that were successfully indexed
