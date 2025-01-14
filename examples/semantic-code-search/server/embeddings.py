import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from together import Together
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
import logging
import time
from functools import wraps

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

class CodeEmbeddings:
    def __init__(self):
        try:
            logger.info("Initializing CodeEmbeddings...")
            
            # Initialize the code embeddings model
            model_name = 'jinaai/jina-embeddings-v2-base-code'
            logger.info(f"Loading model: {model_name}")
            
            # Pre-download the model with trust_remote_code=True
            _ = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            _ = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Now initialize SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully")
            
            # Initialize ChromaDB
            db_path = Path(__file__).parent / "chroma_db"
            db_path.mkdir(exist_ok=True, parents=True)
            db_path = str(db_path.absolute())
            logger.info(f"Initializing ChromaDB at absolute path: {db_path}")
            
            try:
                settings = Settings(
                    persist_directory=db_path,
                    is_persistent=True,
                    anonymized_telemetry=False
                )
                logger.info(f"ChromaDB settings: {settings}")
                
                self.db = chromadb.PersistentClient(
                    path=db_path,
                    settings=settings
                )
                logger.info("ChromaDB client created")
                
                # List all collections
                collections = self.db.list_collections()
                logger.info(f"Existing collections: {collections}")
                
                self.collection = self.db.get_or_create_collection(
                    name="code_snippets",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB collection initialized")
                
                # Log collection info
                collection_count = len(self.collection.get()['ids']) if self.collection.get() else 0
                logger.info(f"Collection has {collection_count} items")
                
            except Exception as e:
                logger.error(f"Error initializing ChromaDB: {str(e)}")
                raise
            
            # Initialize Together client
            self.together_client = Together(api_key="4cf7a223bdb6c857677ce31559591cf19cfa59a2460dcc876de54e9bcc5b68c2")
            logger.info("Together client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize CodeEmbeddings: {str(e)}")
            raise
    
    @log_execution_time
    def generate_docstring(self, code_snippet: str) -> str:
        """Generate a concise docstring for a given code snippet using Together AI."""
        try:
            logger.info("Generating docstring for code snippet")
            response = self.together_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert. Generate a concise 2-3 line docstring for the following code snippet. Focus only on what the code does, not implementation details."},
                    {"role": "user", "content": code_snippet}
                ],
                max_tokens=100,
                temperature=0.3,
                timeout=30
            )
            docstring = response.choices[0].message.content.strip()
            logger.info("Docstring generated successfully")
            return docstring
            
        except Exception as e:
            logger.error(f"Error generating docstring: {str(e)}")
            return "No docstring available"

    @log_execution_time
    def embed_code(self, code_snippet: str) -> List[float]:
        """Generate embeddings for a code snippet."""
        try:
            logger.info("Generating embeddings for code snippet")
            embeddings = self.model.encode(code_snippet).tolist()
            logger.info("Embeddings generated successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    @log_execution_time
    def store_code_snippet(self, 
                          snippet_id: str,
                          code_snippet: str, 
                          metadata: Dict[str, Any]) -> None:
        """Store a code snippet with its embeddings and metadata in the vector database."""
        try:
            logger.info(f"Storing code snippet with ID: {snippet_id}")
            
            # Generate docstring
            docstring = self.generate_docstring(code_snippet)
            logger.info("Docstring generated")
            
            # Generate embeddings
            embeddings = self.embed_code(code_snippet)
            logger.info("Embeddings generated")
            
            # Update metadata with docstring
            metadata['docstring'] = docstring
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=[snippet_id],
                embeddings=[embeddings],
                metadatas=[metadata],
                documents=[code_snippet]
            )
            logger.info(f"Successfully stored code snippet: {metadata.get('name', 'unnamed')}")
            
        except Exception as e:
            logger.error(f"Failed to store code snippet {snippet_id}: {str(e)}")
            raise

    @log_execution_time
    def search_similar_code(self, 
                          query: str, 
                          n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar code snippets using the query."""
        try:
            logger.info(f"Searching for code similar to query: {query[:50]}...")
            
            # Generate query embeddings
            query_embedding = self.embed_code(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'code': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
                
            logger.info(f"Found {len(formatted_results)} similar code snippets")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching for similar code: {str(e)}")
            raise

    @log_execution_time
    def verify_embeddings(self) -> Dict[str, Any]:
        """Verify the integrity of stored embeddings and return statistics."""
        try:
            logger.info("Starting embeddings verification...")
            
            # Get all stored items
            all_ids = self.collection.get()
            if not all_ids or 'ids' not in all_ids:
                logger.warning("No embeddings found in the collection")
                return {
                    'total_items': 0,
                    'valid_items': 0,
                    'items_with_docstrings': 0,
                    'unique_files': 0,
                    'embedding_dimensions': [],
                    'file_paths': []
                }
                
            total_items = len(all_ids['ids'])
            logger.info(f"Found {total_items} stored items")
            logger.debug(f"ChromaDB response structure: {all_ids.keys()}")
            
            # Verify each item has required components
            valid_items = 0
            items_with_docstrings = 0
            embedding_dimensions = set()
            file_paths = set()
            
            for i in range(total_items):
                try:
                    item_id = all_ids['ids'][i]
                    # Get individual item details
                    item = self.collection.get(
                        ids=[item_id],
                        include=['embeddings', 'documents', 'metadatas']
                    )
                    
                    if not item or not item['ids']:
                        logger.warning(f"Could not retrieve item {item_id}")
                        continue
                        
                    # Check embeddings
                    if item['embeddings'] and item['embeddings'][0]:
                        embedding_dimensions.add(len(item['embeddings'][0]))
                    
                    # Check metadata
                    if item['metadatas'] and item['metadatas'][0]:
                        metadata = item['metadatas'][0]
                        if metadata.get('docstring'):
                            items_with_docstrings += 1
                        if metadata.get('file_path'):
                            file_paths.add(metadata['file_path'])
                    
                    # Check document exists
                    if item['documents'] and item['documents'][0]:
                        valid_items += 1
                        
                except Exception as e:
                    logger.error(f"Error verifying item {all_ids['ids'][i]}: {str(e)}")
                    
            # Prepare statistics
            stats = {
                'total_items': total_items,
                'valid_items': valid_items,
                'items_with_docstrings': items_with_docstrings,
                'unique_files': len(file_paths),
                'embedding_dimensions': list(embedding_dimensions),
                'file_paths': list(file_paths)
            }
            
            logger.info("Embeddings verification completed")
            logger.info(f"Statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during embeddings verification: {str(e)}")
            raise

    @log_execution_time
    def get_sample_items(self, n_samples: int = 3) -> List[Dict[str, Any]]:
        """Get a sample of stored items for manual verification."""
        try:
            logger.info(f"Retrieving {n_samples} sample items...")
            
            # Get all items
            all_ids = self.collection.get()
            if not all_ids or 'ids' not in all_ids or not all_ids['ids']:
                logger.warning("No items found in the database")
                return []
            
            total_items = len(all_ids['ids'])
            
            # Get sample indices
            import random
            sample_indices = random.sample(range(total_items), min(n_samples, total_items))
            
            # Get sample items
            samples = []
            for idx in sample_indices:
                item_id = all_ids['ids'][idx]
                # Get individual item details
                item = self.collection.get(
                    ids=[item_id],
                    include=['embeddings', 'documents', 'metadatas']
                )
                
                if not item or not item['ids']:
                    logger.warning(f"Could not retrieve sample item {item_id}")
                    continue
                    
                sample = {
                    'id': item['ids'][0],
                    'code': item['documents'][0] if item['documents'] else None,
                    'metadata': item['metadatas'][0] if item['metadatas'] else {},
                    'embedding_length': len(item['embeddings'][0]) if item['embeddings'] else 0
                }
                samples.append(sample)
            
            logger.info(f"Retrieved {len(samples)} sample items")
            return samples
            
        except Exception as e:
            logger.error(f"Error retrieving sample items: {str(e)}")
            raise
