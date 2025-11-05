"""
ChromaDB Vector Database Module
Handles storage and retrieval of embeddings
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from src.config.settings import Config
from src.config.logger import app_logger
import numpy as np

class ChromaDBManager:
    """Manage ChromaDB operations for vector storage"""
    
    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """
        Initialize ChromaDB client
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist database
        """
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or Config.CHROMA_DB_PATH
        
        app_logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        app_logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Add documents with embeddings to the database
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "title": chunk["metadata"].get("title", ""),
                "author": chunk["metadata"].get("author", ""),
                "lembaga": chunk["metadata"].get("lembaga", "Unknown"),  # Tambahkan info lembaga
            }
            for chunk in chunks
        ]
        
        # Convert embeddings to list format
        embeddings_list = embeddings.tolist()
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                embeddings=embeddings_list[i:batch_end]
            )
            
            app_logger.info(f"Added batch {i//batch_size + 1}: {batch_end - i} documents")
        
        app_logger.info(f"Successfully added {len(chunks)} documents to ChromaDB")
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = None,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Query the database with an embedding
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_dict: Metadata filter dictionary
            
        Returns:
            Query results dictionary
        """
        n_results = n_results or Config.TOP_K_RETRIEVAL
        
        # Convert embedding to list
        query_embedding_list = query_embedding.tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            where=filter_dict
        )
        
        return results
    
    def search(
        self,
        query_text: str,
        n_results: int = None,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Search the database with a text query
        (Alternative method for compatibility with Self-RAG pipeline)
        
        Args:
            query_text: Query text string
            n_results: Number of results to return
            filter_dict: Metadata filter dictionary
            
        Returns:
            Query results dictionary with embeddings attribute
        """
        from src.embeddings.indobert_embeddings import get_embedding_model
        
        n_results = n_results or Config.TOP_K_RETRIEVAL
        
        # Get embedding model and generate query embedding
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_text(query_text)
        
        # Use the existing query method
        results = self.query(
            query_embedding=query_embedding,
            n_results=n_results,
            filter_dict=filter_dict
        )
        
        # Add embeddings attribute for compatibility
        results['embeddings'] = self.embeddings if hasattr(self, 'embeddings') else None
        
        return results
    
    def delete_by_source(self, source_file: str):
        """
        Delete all documents from a specific source file
        
        Args:
            source_file: Name of the source file
        """
        results = self.collection.get(
            where={"source_file": source_file}
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            app_logger.info(f"Deleted {len(results['ids'])} documents from {source_file}")
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        app_logger.info(f"Cleared collection '{self.collection_name}'")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        # Get unique source files
        all_docs = self.collection.get()
        source_files = set(meta.get("source_file", "") for meta in all_docs["metadatas"])
        
        return {
            "total_documents": count,
            "unique_sources": len(source_files),
            "collection_name": self.collection_name
        }

# Alias for compatibility
ChromaManager = ChromaDBManager

# Singleton instance
_db_instance = None

def get_vector_db() -> ChromaDBManager:
    """
    Get or create singleton ChromaDB instance
    
    Returns:
        ChromaDBManager instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = ChromaDBManager()
    return _db_instance
