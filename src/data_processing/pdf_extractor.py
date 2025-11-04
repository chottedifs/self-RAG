"""
PDF Extraction Module
Extracts text content from PDF files
"""
from typing import Dict, List
from pathlib import Path
import fitz  # PyMuPDF
from src.config.logger import app_logger
from src.utils.helpers import compute_file_hash

class PDFExtractor:
    """Extract text content from PDF documents"""
    
    def __init__(self):
        """Initialize PDF extractor"""
        pass
    
    def extract_from_file(self, pdf_path: Path, parent_folder: str = None) -> Dict[str, any]:
        """
        Extract text from a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            parent_folder: Name of parent folder (e.g., nama lembaga)
            
        Returns:
            Dictionary containing extracted information
        """
        app_logger.info(f"Extracting text from: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                "filename": pdf_path.name,
                "filepath": str(pdf_path),
                "file_hash": compute_file_hash(pdf_path),
                "num_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "lembaga": parent_folder or "Unknown",  # Tambahkan info lembaga
            }
            
            # Extract text from all pages
            pages_text = []
            full_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                pages_text.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text)
                })
                
                full_text.append(text)
            
            doc.close()
            
            result = {
                "metadata": metadata,
                "pages": pages_text,
                "full_text": "\n\n".join(full_text),
                "total_chars": sum(p["char_count"] for p in pages_text)
            }
            
            app_logger.info(f"Extracted {len(pages_text)} pages, {result['total_chars']} characters")
            
            return result
            
        except Exception as e:
            app_logger.error(f"Failed to extract PDF {pdf_path.name}: {e}")
            raise
    
    def extract_from_directory(self, directory: Path, recursive: bool = True) -> List[Dict]:
        """
        Extract text from all PDF files in a directory
        
        Args:
            directory: Path to directory containing PDFs
            recursive: If True, search in subdirectories (for lembaga folders)
            
        Returns:
            List of extraction results
        """
        results = []
        
        if recursive:
            # Search recursively in subdirectories (e.g., lembaga folders)
            pdf_files = list(directory.rglob("*.pdf"))
            app_logger.info(f"Found {len(pdf_files)} PDF files recursively in {directory}")
            
            for pdf_file in pdf_files:
                try:
                    # Get parent folder name (nama lembaga)
                    relative_path = pdf_file.relative_to(directory)
                    parent_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else None
                    
                    app_logger.info(f"Processing: {pdf_file.name} from lembaga: {parent_folder or 'root'}")
                    
                    result = self.extract_from_file(pdf_file, parent_folder)
                    results.append(result)
                except Exception as e:
                    app_logger.error(f"Skipping {pdf_file.name} due to error: {e}")
                    continue
        else:
            # Search only in current directory (non-recursive)
            pdf_files = list(directory.glob("*.pdf"))
            app_logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
            
            for pdf_file in pdf_files:
                try:
                    result = self.extract_from_file(pdf_file)
                    results.append(result)
                except Exception as e:
                    app_logger.error(f"Skipping {pdf_file.name} due to error: {e}")
                    continue
        
        return results
