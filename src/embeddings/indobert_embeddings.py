"""
IndoBERT Embedding Module
Handles text embedding using IndoBERT model
"""
from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from src.config.settings import Config
from src.config.logger import app_logger

class IndoBERTEmbedding:
    """IndoBERT-based embedding generator"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize IndoBERT embedding model
        
        Args:
            model_name: HuggingFace model name (default: from config)
        """
        self.model_name = model_name or Config.INDOBERT_MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        app_logger.info(f"Loading IndoBERT model: {self.model_name}")
        app_logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            app_logger.info("IndoBERT model loaded successfully")
        except Exception as e:
            app_logger.error(f"Failed to load IndoBERT model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embeddings
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Generate embeddings
                outputs = self.model(**encoded)
                
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings
        
        Returns:
            Embedding dimension
        """
        return self.model.config.hidden_size
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Callable interface for embedding generation
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Embeddings array
        """
        if isinstance(text, str):
            return self.embed_text(text)
        else:
            return self.embed_texts(text)

# Singleton instance
_embedding_instance = None

def get_embedding_model() -> IndoBERTEmbedding:
    """
    Get or create singleton embedding model instance
    
    Returns:
        IndoBERTEmbedding instance
    """
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = IndoBERTEmbedding()
    return _embedding_instance
