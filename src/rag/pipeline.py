"""
RAG (Retrieval-Augmented Generation) Pipeline
Combines retrieval and generation for question answering
"""
from typing import List, Dict, Optional
import numpy as np

from src.config.settings import Config
from src.config.logger import app_logger
from src.embeddings.indobert_embeddings import get_embedding_model
from src.vector_db.chroma_manager import get_vector_db
from src.llm.ollama_client import get_ollama_llm
from src.utils.helpers import format_context

class RAGPipeline:
    """RAG pipeline for question answering"""
    
    def __init__(self):
        """Initialize RAG pipeline components"""
        app_logger.info("Initializing RAG Pipeline")
        
        self.embedding_model = get_embedding_model()
        self.vector_db = get_vector_db()
        self.llm = get_ollama_llm()
        
        # RAG configuration
        self.top_k = Config.TOP_K_RETRIEVAL
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        
        app_logger.info("RAG Pipeline ready")
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_dict: Metadata filter
            
        Returns:
            List of retrieved document dictionaries
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        app_logger.info(f"Generating embedding for query: {query[:100]}...")
        query_embedding = self.embedding_model.embed_text(query)
        
        # Retrieve from vector database
        app_logger.info(f"Retrieving top {top_k} relevant documents...")
        results = self.vector_db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            filter_dict=filter_dict
        )
        
        # Format results
        retrieved_docs = []
        
        if results["ids"] and len(results["ids"][0]) > 0:
            for idx, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity score (for cosine distance)
                similarity = 1 - distance
                
                if similarity >= self.similarity_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity": similarity,
                        "rank": idx + 1
                    })
        
        app_logger.info(f"Retrieved {len(retrieved_docs)} documents above threshold")
        
        return retrieved_docs
    
    def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate a response using retrieved context
        
        Args:
            query: User query
            context_docs: Retrieved context documents
            temperature: LLM temperature
            
        Returns:
            Dictionary with response and metadata
        """
        # Prepare context
        if not context_docs:
            context_text = "Tidak ada konteks yang relevan ditemukan."
        else:
            context_chunks = [doc["content"] for doc in context_docs]
            context_text = format_context(context_chunks)
        
        # Create system prompt
        system_prompt = """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.
Gunakan konteks yang diberikan untuk menjawab pertanyaan dengan akurat dan informatif.
Jika informasi tidak tersedia dalam konteks, katakan dengan jujur bahwa Anda tidak memiliki informasi tersebut.
Jawab dalam Bahasa Indonesia yang baik dan benar."""
        
        # Create user prompt with context
        user_prompt = f"""Konteks:
{context_text}

Pertanyaan: {query}

Jawaban:"""
        
        # Generate response
        app_logger.info("Generating response with LLM...")
        response_text = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # Prepare response with metadata
        response = {
            "answer": response_text.strip(),
            "query": query,
            "num_sources": len(context_docs),
            "sources": [
                {
                    "file": doc["metadata"].get("source_file", "Unknown"),
                    "lembaga": doc["metadata"].get("lembaga", "Unknown"),  # Tambahkan info lembaga
                    "similarity": doc["similarity"],
                    "rank": doc["rank"]
                }
                for doc in context_docs
            ]
        }
        
        return response
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Complete RAG query: retrieve context and generate response
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            filter_dict: Metadata filter
            
        Returns:
            Response dictionary
        """
        app_logger.info(f"RAG Query: {query[:100]}...")
        
        # Step 1: Retrieve relevant context
        context_docs = self.retrieve_context(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        # Step 2: Generate response
        response = self.generate_response(
            query=query,
            context_docs=context_docs,
            temperature=temperature
        )
        
        app_logger.info("RAG query completed successfully")
        
        return response

# Singleton instance
_rag_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create singleton RAG pipeline instance
    
    Returns:
        RAGPipeline instance
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance
