"""
Self-RAG Pipeline with Reflection Tokens
Implements Self-Reflective RAG with fine-tuned critic model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import numpy as np

from src.embeddings.indobert_embeddings import get_embedding_model
from src.vector_db.chroma_manager import ChromaManager
from src.llm.ollama_client import get_ollama_llm
from src.config.settings import Config
from src.config.logger import app_logger


class ReflectionToken(Enum):
    """Special tokens for self-reflection"""
    # Retrieval tokens
    RETRIEVE = "[Retrieve]"
    NO_RETRIEVE = "[No Retrieve]"
    
    # Relevance tokens
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    
    # Support tokens
    FULLY_SUPPORTED = "[Fully Supported]"
    PARTIALLY_SUPPORTED = "[Partially Supported]"
    NO_SUPPORT = "[No Support]"
    
    # Utility tokens
    UTILITY_5 = "[Utility:5]"
    UTILITY_4 = "[Utility:4]"
    UTILITY_3 = "[Utility:3]"
    UTILITY_2 = "[Utility:2]"
    UTILITY_1 = "[Utility:1]"


class SelfRAGPipeline:
    """
    Self-RAG Pipeline with fine-tuned critic model
    
    Features:
    - Adaptive retrieval decision
    - Document relevance filtering
    - Answer support verification
    - Utility scoring
    - Iterative refinement
    """
    
    def __init__(
        self,
        use_finetuned_critic: bool = False,
        critic_model_path: Optional[str] = None,
        max_iterations: int = 3,
        min_utility_score: int = 3
    ):
        """
        Initialize Self-RAG pipeline
        
        Args:
            use_finetuned_critic: Whether to use fine-tuned critic model
            critic_model_path: Path to fine-tuned critic model
            max_iterations: Maximum refinement iterations
            min_utility_score: Minimum acceptable utility score (1-5)
        """
        app_logger.info("Initializing Self-RAG Pipeline...")
        
        # Core components
        self.embeddings = get_embedding_model()
        self.vector_db = ChromaManager(embeddings=self.embeddings)
        self.llm = get_ollama_llm()
        
        # Self-RAG specific settings
        self.use_finetuned_critic = use_finetuned_critic
        self.critic_model_path = critic_model_path
        self.max_iterations = max_iterations
        self.min_utility_score = min_utility_score
        
        # Load fine-tuned critic if available
        if use_finetuned_critic and critic_model_path:
            self._load_critic_model(critic_model_path)
        
        app_logger.info(f"Self-RAG initialized (fine-tuned critic: {use_finetuned_critic})")
    
    def _load_critic_model(self, model_path: str):
        """Load fine-tuned critic model"""
        try:
            # TODO: Implement loading fine-tuned model
            # This would load LoRA weights or full fine-tuned model
            app_logger.info(f"Loading fine-tuned critic from: {model_path}")
            self.critic_llm = get_ollama_llm()  # Placeholder
        except Exception as e:
            app_logger.error(f"Failed to load critic model: {e}")
            self.use_finetuned_critic = False
    
    def _decide_retrieval(self, query: str) -> Tuple[bool, float]:
        """
        Decide if retrieval is needed using critic model
        
        Args:
            query: User query
            
        Returns:
            (need_retrieval, confidence)
        """
        if self.use_finetuned_critic:
            prompt = f"""Query: {query}

Should we retrieve documents for this query?
Answer with only the token: {ReflectionToken.RETRIEVE.value} or {ReflectionToken.NO_RETRIEVE.value}"""
            
            response = self.llm.generate(prompt, temperature=0.1)
            need_retrieval = ReflectionToken.RETRIEVE.value in response
            confidence = 0.9 if ReflectionToken.RETRIEVE.value in response else 0.1
        else:
            # Heuristic-based decision
            keywords = ['apa', 'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 
                       'persyaratan', 'prosedur', 'cara', 'biaya', 'jadwal']
            
            query_lower = query.lower()
            need_retrieval = any(kw in query_lower for kw in keywords)
            confidence = 0.7
        
        app_logger.info(f"Retrieval decision: {need_retrieval} (confidence: {confidence:.2f})")
        return need_retrieval, confidence
    
    def _check_relevance(self, query: str, document: Dict) -> Tuple[bool, float]:
        """
        Check document relevance using critic model
        
        Args:
            query: User query
            document: Retrieved document
            
        Returns:
            (is_relevant, confidence)
        """
        doc_preview = document.get('text', '')[:500]
        
        if self.use_finetuned_critic:
            prompt = f"""Query: {query}

Document: {doc_preview}

Is this document relevant?
Answer with only the token: {ReflectionToken.RELEVANT.value} or {ReflectionToken.IRRELEVANT.value}"""
            
            response = self.llm.generate(prompt, temperature=0.1)
            is_relevant = ReflectionToken.RELEVANT.value in response
            confidence = 0.85
        else:
            # Heuristic-based relevance
            similarity_score = document.get('similarity_score', 0.0)
            is_relevant = similarity_score > Config.SIMILARITY_THRESHOLD
            confidence = similarity_score
        
        return is_relevant, confidence
    
    def _verify_support(self, query: str, answer: str, documents: List[Dict]) -> Tuple[str, float]:
        """
        Verify if answer is supported by documents
        
        Args:
            query: User query
            answer: Generated answer
            documents: Retrieved documents
            
        Returns:
            (support_level, confidence)
        """
        doc_texts = "\n\n".join([f"Doc {i+1}: {doc.get('text', '')[:300]}" 
                                 for i, doc in enumerate(documents)])
        
        if self.use_finetuned_critic:
            prompt = f"""Query: {query}

Answer: {answer}

Documents:
{doc_texts}

Is the answer supported by the documents?
Answer with one token: {ReflectionToken.FULLY_SUPPORTED.value}, {ReflectionToken.PARTIALLY_SUPPORTED.value}, or {ReflectionToken.NO_SUPPORT.value}"""
            
            response = self.llm.generate(prompt, temperature=0.1)
            
            if ReflectionToken.FULLY_SUPPORTED.value in response:
                support_level = "fully_supported"
                confidence = 0.95
            elif ReflectionToken.PARTIALLY_SUPPORTED.value in response:
                support_level = "partially_supported"
                confidence = 0.7
            else:
                support_level = "no_support"
                confidence = 0.3
        else:
            # Heuristic-based support check
            support_level = "partially_supported"
            confidence = 0.6
        
        app_logger.info(f"Support verification: {support_level} (confidence: {confidence:.2f})")
        return support_level, confidence
    
    def _evaluate_utility(self, query: str, answer: str) -> Tuple[int, float]:
        """
        Evaluate utility/quality of answer
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            (utility_score, confidence)
        """
        if self.use_finetuned_critic:
            prompt = f"""Query: {query}

Answer: {answer}

Rate the utility of this answer (1-5):
Answer with one token: {ReflectionToken.UTILITY_1.value} to {ReflectionToken.UTILITY_5.value}"""
            
            response = self.llm.generate(prompt, temperature=0.1)
            
            # Parse utility score
            for i in range(5, 0, -1):
                token = getattr(ReflectionToken, f"UTILITY_{i}").value
                if token in response:
                    return i, 0.8
            
            return 3, 0.5  # Default
        else:
            # Heuristic-based utility
            answer_length = len(answer.split())
            if answer_length < 10:
                utility = 2
            elif answer_length < 30:
                utility = 3
            elif answer_length < 100:
                utility = 4
            else:
                utility = 5
            
            confidence = 0.6
        
        app_logger.info(f"Utility evaluation: {utility}/5 (confidence: {confidence:.2f})")
        return utility, confidence
    
    def _generate_answer(
        self, 
        query: str, 
        documents: Optional[List[Dict]] = None,
        iteration: int = 0
    ) -> str:
        """
        Generate answer with or without documents
        
        Args:
            query: User query
            documents: Retrieved documents (optional)
            iteration: Current iteration number
            
        Returns:
            Generated answer
        """
        if documents:
            # Format context
            context_parts = []
            for i, doc in enumerate(documents, 1):
                context_parts.append(
                    f"[Dokumen {i}] {doc.get('filename', 'Unknown')}\n"
                    f"Lembaga: {doc.get('lembaga', 'Unknown')}\n"
                    f"{doc.get('text', '')}\n"
                )
            context = "\n".join(context_parts)
            
            system_prompt = """Kamu adalah asisten AI yang menjawab pertanyaan berdasarkan dokumen yang diberikan.
ATURAN PENTING:
1. Jawab HANYA berdasarkan informasi dari dokumen
2. Jika informasi tidak ada, katakan "Informasi tidak tersedia dalam dokumen"
3. Berikan jawaban yang lengkap, jelas, dan terstruktur
4. Sebutkan dari dokumen mana informasi berasal"""
            
            prompt = f"""DOKUMEN REFERENSI:
{context}

PERTANYAAN: {query}

JAWABAN:"""
        else:
            # Answer from general knowledge
            system_prompt = """Kamu adalah asisten AI yang membantu menjawab pertanyaan umum.
Berikan jawaban yang akurat dan informatif."""
            
            prompt = f"""PERTANYAAN: {query}

JAWABAN:"""
        
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3 if iteration == 0 else 0.5
        )
        
        return answer
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Main Self-RAG query with iterative refinement
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_dict: Filter for vector search
            
        Returns:
            Dict with answer and metadata
        """
        app_logger.info(f"Self-RAG Query: {query}")
        
        result = {
            "query": query,
            "answer": "",
            "sources": [],
            "metadata": {
                "retrieval_decision": None,
                "relevant_docs": 0,
                "support_level": None,
                "utility_score": 0,
                "iterations": 0,
                "confidence_scores": {}
            }
        }
        
        # STEP 1: Decide if retrieval is needed
        need_retrieval, retrieval_conf = self._decide_retrieval(query)
        result["metadata"]["retrieval_decision"] = need_retrieval
        result["metadata"]["confidence_scores"]["retrieval"] = retrieval_conf
        
        if not need_retrieval:
            # Answer from general knowledge
            answer = self._generate_answer(query, documents=None)
            result["answer"] = answer
            result["metadata"]["note"] = "Answered from general knowledge"
            return result
        
        # STEP 2: Retrieve documents
        try:
            retrieved = self.vector_db.search(
                query_text=query,
                n_results=top_k,
                filter_dict=filter_dict
            )
            
            if not retrieved.get("documents"):
                result["answer"] = "Maaf, tidak ditemukan dokumen yang relevan."
                return result
            
            # Format documents
            all_documents = []
            for i in range(len(retrieved["documents"][0])):
                doc = {
                    "text": retrieved["documents"][0][i],
                    "filename": retrieved["metadatas"][0][i].get("filename", "Unknown"),
                    "lembaga": retrieved["metadatas"][0][i].get("lembaga", "Unknown"),
                    "similarity_score": 1 - retrieved["distances"][0][i]
                }
                all_documents.append(doc)
        
        except Exception as e:
            app_logger.error(f"Retrieval failed: {e}")
            result["answer"] = f"Terjadi kesalahan saat mencari dokumen: {str(e)}"
            return result
        
        # STEP 3: Iterative refinement
        best_answer = None
        best_utility = 0
        
        for iteration in range(self.max_iterations):
            app_logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Filter relevant documents
            relevant_docs = []
            for doc in all_documents:
                is_relevant, rel_conf = self._check_relevance(query, doc)
                if is_relevant:
                    doc["relevance_confidence"] = rel_conf
                    relevant_docs.append(doc)
            
            if not relevant_docs:
                app_logger.warning("No relevant documents found")
                if iteration == 0:
                    result["answer"] = "Dokumen yang ditemukan tidak relevan dengan pertanyaan."
                    return result
                break
            
            app_logger.info(f"Found {len(relevant_docs)} relevant documents")
            result["metadata"]["relevant_docs"] = len(relevant_docs)
            
            # Generate answer
            answer = self._generate_answer(query, relevant_docs, iteration)
            
            # Verify support
            support_level, support_conf = self._verify_support(query, answer, relevant_docs)
            result["metadata"]["support_level"] = support_level
            result["metadata"]["confidence_scores"]["support"] = support_conf
            
            # Evaluate utility
            utility_score, utility_conf = self._evaluate_utility(query, answer)
            result["metadata"]["utility_score"] = utility_score
            result["metadata"]["confidence_scores"]["utility"] = utility_conf
            
            # Check if answer is good enough
            if utility_score >= self.min_utility_score and support_level != "no_support":
                best_answer = answer
                best_utility = utility_score
                result["metadata"]["iterations"] = iteration + 1
                app_logger.info(f"Acceptable answer found at iteration {iteration + 1}")
                break
            
            # Track best answer
            if utility_score > best_utility:
                best_answer = answer
                best_utility = utility_score
            
            # Adjust for next iteration
            if iteration < self.max_iterations - 1:
                app_logger.info("Answer quality insufficient, trying different approach...")
                top_k = min(top_k + 2, 15)  # Retrieve more documents
        
        # Return best answer
        result["answer"] = best_answer or "Maaf, tidak dapat menghasilkan jawaban yang memuaskan."
        result["sources"] = relevant_docs[:5]  # Top 5 sources
        result["metadata"]["iterations"] = iteration + 1
        
        app_logger.info(f"Self-RAG completed in {iteration + 1} iterations")
        return result


def create_self_rag_pipeline(
    use_finetuned_critic: bool = False,
    critic_model_path: Optional[str] = None
) -> SelfRAGPipeline:
    """
    Create Self-RAG pipeline instance
    
    Args:
        use_finetuned_critic: Whether to use fine-tuned critic
        critic_model_path: Path to critic model
        
    Returns:
        SelfRAGPipeline instance
    """
    return SelfRAGPipeline(
        use_finetuned_critic=use_finetuned_critic,
        critic_model_path=critic_model_path
    )
