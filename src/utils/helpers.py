"""
Utility functions for the RAG Chatbot
"""
import hashlib
from typing import List, Any
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Ensure chunk ends at word boundary if possible
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # Only adjust if space is near the end
                end = start + last_space
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def format_context(chunks: List[str]) -> str:
    """
    Format retrieved chunks into a single context string
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Formatted context string
    """
    formatted = "\n\n---\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    return formatted

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be safe for use in filesystem
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return sanitized
