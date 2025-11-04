"""
Text Chunking Module
Splits extracted text into manageable chunks
"""
from typing import List, Dict
from src.config.settings import Config
from src.config.logger import app_logger
from src.utils.helpers import chunk_text

class TextChunker:
    """Chunk text into smaller pieces for embedding"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.overlap = overlap or Config.CHUNK_OVERLAP
        
        app_logger.info(f"TextChunker initialized: chunk_size={self.chunk_size}, overlap={self.overlap}")
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a document into smaller pieces
        
        Args:
            document: Document dictionary from PDFExtractor
            
        Returns:
            List of chunk dictionaries
        """
        full_text = document["full_text"]
        metadata = document["metadata"]
        
        # Create text chunks
        text_chunks = chunk_text(full_text, self.chunk_size, self.overlap)
        
        # Create chunk dictionaries with metadata
        chunks = []
        for idx, text_chunk in enumerate(text_chunks):
            chunk = {
                "chunk_id": f"{metadata['file_hash']}_{idx}",
                "chunk_index": idx,
                "text": text_chunk,
                "char_count": len(text_chunk),
                "source_file": metadata["filename"],
                "source_hash": metadata["file_hash"],
                "metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "num_pages": metadata.get("num_pages", 0),
                    "lembaga": metadata.get("lembaga", "Unknown"),  # Tambahkan info lembaga
                }
            }
            chunks.append(chunk)
        
        app_logger.info(f"Created {len(chunks)} chunks from {metadata['filename']}")
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for document in documents:
            try:
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
            except Exception as e:
                app_logger.error(f"Failed to chunk document: {e}")
                continue
        
        app_logger.info(f"Total chunks created: {len(all_chunks)}")
        
        return all_chunks
