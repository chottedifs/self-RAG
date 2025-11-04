"""
Data Preparation Pipeline
Orchestrates the entire data preparation process
"""
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import Config
from src.config.logger import app_logger
from src.data_processing.pdf_extractor import PDFExtractor
from src.data_processing.text_chunker import TextChunker
from src.embeddings.indobert_embeddings import get_embedding_model
from src.vector_db.chroma_manager import get_vector_db

class DataPreparationPipeline:
    """Complete pipeline for preparing and indexing PDF documents"""
    
    def __init__(self):
        """Initialize pipeline components"""
        app_logger.info("Initializing Data Preparation Pipeline")
        
        self.pdf_extractor = PDFExtractor()
        self.text_chunker = TextChunker()
        self.embedding_model = None  # Lazy load
        self.vector_db = None  # Lazy load
    
    def _ensure_models_loaded(self):
        """Lazy load models to avoid loading them unnecessarily"""
        if self.embedding_model is None:
            app_logger.info("Loading embedding model...")
            self.embedding_model = get_embedding_model()
        
        if self.vector_db is None:
            app_logger.info("Initializing vector database...")
            self.vector_db = get_vector_db()
    
    def process_pdf_directory(self, clear_existing: bool = False):
        """
        Process all PDFs in the data resources directory
        
        Args:
            clear_existing: Whether to clear existing data before processing
        """
        data_dir = Path(Config.DATA_RESOURCES_DIR)
        
        app_logger.info(f"Starting data preparation from: {data_dir}")
        
        # Step 1: Extract text from PDFs
        app_logger.info("Step 1/4: Extracting text from PDFs...")
        documents = self.pdf_extractor.extract_from_directory(data_dir)
        
        if not documents:
            app_logger.warning("No PDF documents found!")
            return
        
        app_logger.info(f"Extracted {len(documents)} documents")
        
        # Step 2: Chunk documents
        app_logger.info("Step 2/4: Chunking documents...")
        all_chunks = self.text_chunker.chunk_documents(documents)
        
        if not all_chunks:
            app_logger.warning("No chunks created!")
            return
        
        app_logger.info(f"Created {len(all_chunks)} chunks")
        
        # Load models
        self._ensure_models_loaded()
        
        # Clear existing data if requested
        if clear_existing:
            app_logger.info("Clearing existing vector database...")
            self.vector_db.clear_collection()
        
        # Step 3: Generate embeddings
        app_logger.info("Step 3/4: Generating embeddings with IndoBERT...")
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        
        # Generate embeddings with progress bar
        embeddings = self.embedding_model.embed_texts(chunk_texts)
        
        app_logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Step 4: Store in vector database
        app_logger.info("Step 4/4: Storing in ChromaDB...")
        self.vector_db.add_documents(all_chunks, embeddings)
        
        # Show statistics
        stats = self.vector_db.get_stats()
        app_logger.info(f"Pipeline complete! Stats: {stats}")
        
        return stats
    
    def process_single_pdf(self, pdf_path: Path):
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
        """
        app_logger.info(f"Processing single PDF: {pdf_path.name}")
        
        # Extract
        document = self.pdf_extractor.extract_from_file(pdf_path)
        
        # Chunk
        chunks = self.text_chunker.chunk_document(document)
        
        # Load models
        self._ensure_models_loaded()
        
        # Embed
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(chunk_texts)
        
        # Store
        self.vector_db.add_documents(chunks, embeddings)
        
        app_logger.info(f"Successfully processed {pdf_path.name}")

def main():
    """Main entry point for data preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare PDF data for RAG system")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before processing"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process a single PDF file"
    )
    
    args = parser.parse_args()
    
    pipeline = DataPreparationPipeline()
    
    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            app_logger.error(f"File not found: {pdf_path}")
            return
        pipeline.process_single_pdf(pdf_path)
    else:
        pipeline.process_pdf_directory(clear_existing=args.clear)

if __name__ == "__main__":
    main()
