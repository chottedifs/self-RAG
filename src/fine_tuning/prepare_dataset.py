"""
Dataset Preparation for Self-RAG Fine-tuning

Creates training data for critic model with reflection tokens
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path

from src.vector_db.chroma_manager import get_vector_db
from src.embeddings.indobert_embeddings import get_embedding_model
from src.llm.ollama_client import get_ollama_llm
from src.config.logger import app_logger


class SelfRAGDatasetGenerator:
    """Generate training dataset for Self-RAG critic model using real documents"""
    
    def __init__(self, output_dir: str = "./fine_tuning_data"):
        """
        Initialize dataset generator
        
        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embeddings = get_embedding_model()
        self.vector_db = get_vector_db()
        self.llm = get_ollama_llm()
        
        app_logger.info(f"Dataset generator initialized (output: {output_dir})")
        app_logger.info("📚 Using real documents from ChromaDB to generate training data")
    
    def _generate_retrieval_examples(self, num_examples: int = 100) -> List[Dict]:
        """
        Generate retrieval decision examples from real documents
        """
        examples = []
        
        try:
            # Get all documents from vector DB
            all_docs = self.vector_db.collection.get()
            
            if not all_docs.get("documents"):
                app_logger.warning("No documents in vector DB, using fallback queries")
                # Fallback to basic queries
                need_retrieval = [
                    "Apa persyaratan beasiswa?",
                    "Bagaimana prosedur pendaftaran?",
                    "Apa saja program yang tersedia?"
                ]
                no_retrieval = [
                    "Siapa presiden Indonesia?",
                    "Berapa hasil 10 + 20?",
                    "Apa ibukota Jakarta?"
                ]
            else:
                # Extract topics from real documents
                need_retrieval = []
                seen_topics = set()
                
                for metadata in all_docs.get("metadatas", []):
                    filename = metadata.get("filename", "")
                    lembaga = metadata.get("lembaga", "")
                    
                    # Generate queries based on document filenames/topics
                    if filename and filename not in seen_topics:
                        seen_topics.add(filename)
                        
                        # Create domain-specific queries
                        if "SISKERMA" in filename.upper():
                            need_retrieval.extend([
                                "Bagaimana cara login ke SISKERMA?",
                                "Apa fungsi SISKERMA?",
                                "Bagaimana cara mengakses SISKERMA?"
                            ])
                        elif "BEASISWA" in filename.upper():
                            need_retrieval.extend([
                                "Apa persyaratan beasiswa?",
                                "Bagaimana cara mendaftar beasiswa?",
                                "Kapan pendaftaran beasiswa dibuka?"
                            ])
                        elif lembaga:
                            need_retrieval.extend([
                                f"Apa tugas {lembaga}?",
                                f"Bagaimana cara menghubungi {lembaga}?",
                                f"Apa program dari {lembaga}?"
                            ])
                        
                        if len(need_retrieval) >= num_examples // 2:
                            break
                
                # General knowledge queries (don't need retrieval)
                no_retrieval = [
                    "Siapa presiden Indonesia?",
                    "Apa ibukota Jawa Barat?",
                    "Kapan Indonesia merdeka?",
                    "Berapa hasil 10 + 20?",
                    "Apa bahasa resmi Indonesia?",
                    "Siapa penemu telepon?",
                    "Apa warna langit?",
                    "Berapa jumlah provinsi di Indonesia?",
                    "Apa nama mata uang Indonesia?",
                    "Siapa penulis novel Laskar Pelangi?"
                ]
            
            # Generate [Retrieve] examples
            for query in need_retrieval[:num_examples//2]:
                examples.append({
                    "instruction": f"Query: {query}\n\nShould we retrieve documents for this query?",
                    "input": "",
                    "output": "[Retrieve]",
                    "category": "retrieval_decision"
                })
            
            # Generate [No Retrieve] examples
            for query in no_retrieval[:num_examples//2]:
                examples.append({
                    "instruction": f"Query: {query}\n\nShould we retrieve documents for this query?",
                    "input": "",
                    "output": "[No Retrieve]",
                    "category": "retrieval_decision"
                })
        
        except Exception as e:
            app_logger.error(f"Error generating retrieval examples: {e}")
        
        app_logger.info(f"Generated {len(examples)} retrieval decision examples from real data")
        return examples
    
    def _generate_relevance_examples(self, num_examples: int = 100) -> List[Dict]:
        """
        Generate document relevance examples using real documents from ChromaDB
        """
        examples = []
        
        try:
            # Get all documents from vector DB
            all_docs = self.vector_db.collection.get()
            
            if not all_docs.get("documents"):
                app_logger.warning("No documents in vector DB")
                return examples
            
            documents = all_docs["documents"]
            metadatas = all_docs["metadatas"]
            
            # Sample documents to use
            indices = list(range(len(documents)))
            random.shuffle(indices)
            
            count = 0
            for idx in indices:
                if count >= num_examples:
                    break
                
                doc_text = documents[idx][:500]  # Use first 500 chars
                metadata = metadatas[idx]
                filename = metadata.get("filename", "Unknown")
                lembaga = metadata.get("lembaga", "")
                
                # Generate RELEVANT query-document pairs
                # Create query based on document content keywords
                if "SISKERMA" in filename.upper() or "SISKERMA" in doc_text.upper():
                    queries = [
                        "Bagaimana cara menggunakan SISKERMA?",
                        "Apa fungsi SISKERMA?",
                        "Bagaimana cara login SISKERMA?"
                    ]
                elif "BEASISWA" in filename.upper() or "BEASISWA" in doc_text.upper():
                    queries = [
                        "Apa persyaratan beasiswa?",
                        "Bagaimana cara mendaftar beasiswa?",
                        "Siapa yang berhak mendapat beasiswa?"
                    ]
                elif "MAHASISWA" in doc_text.upper():
                    queries = [
                        "Apa hak mahasiswa?",
                        "Apa layanan untuk mahasiswa?",
                        "Bagaimana ketentuan mahasiswa?"
                    ]
                elif lembaga:
                    queries = [
                        f"Apa tugas dan fungsi {lembaga}?",
                        f"Bagaimana cara menghubungi {lembaga}?",
                        f"Apa program {lembaga}?"
                    ]
                else:
                    # Extract key terms from document
                    words = doc_text.split()[:20]
                    queries = [f"Informasi tentang {' '.join(words[:5])}?"]
                
                # Add relevant example
                relevant_query = random.choice(queries)
                examples.append({
                    "instruction": f"Query: {relevant_query}\n\nDocument: {doc_text}\n\nIs this document relevant to the query?",
                    "input": "",
                    "output": "[Relevant]",
                    "category": "relevance_check"
                })
                count += 1
                
                # Add irrelevant example (50% of the time)
                if count < num_examples and random.random() < 0.5:
                    irrelevant_queries = [
                        "Siapa presiden Indonesia?",
                        "Berapa jumlah pulau di Indonesia?",
                        "Apa ibukota Jawa Barat?",
                        "Kapan Indonesia merdeka?",
                        "Siapa penemu telepon?"
                    ]
                    irrelevant_query = random.choice(irrelevant_queries)
                    
                    examples.append({
                        "instruction": f"Query: {irrelevant_query}\n\nDocument: {doc_text}\n\nIs this document relevant to the query?",
                        "input": "",
                        "output": "[Irrelevant]",
                        "category": "relevance_check"
                    })
                    count += 1
        
        except Exception as e:
            app_logger.error(f"Failed to generate relevance examples: {e}")
        
        app_logger.info(f"Generated {len(examples)} relevance check examples from real documents")
        return examples
    
    def _generate_support_examples(self, num_examples: int = 100) -> List[Dict]:
        """
        Generate answer support verification examples using real documents
        """
        examples = []
        
        try:
            # Get documents from vector DB
            all_docs = self.vector_db.collection.get()
            
            if not all_docs.get("documents"):
                app_logger.warning("No documents in vector DB for support examples")
                return examples
            
            documents = all_docs["documents"]
            metadatas = all_docs["metadatas"]
            
            # Sample documents
            indices = list(range(len(documents)))
            random.shuffle(indices)
            
            for idx in indices[:num_examples]:
                doc_text = documents[idx][:400]
                metadata = metadatas[idx]
                filename = metadata.get("filename", "Unknown")
                
                # Use LLM to generate query and answer from document
                try:
                    # Generate a query based on document
                    prompt_query = f"Berdasarkan teks berikut, buat 1 pertanyaan yang bisa dijawab dari teks ini:\n\n{doc_text}\n\nPertanyaan:"
                    generated_query = self.llm.generate(prompt_query, max_tokens=100)
                    
                    # Generate answer from document
                    prompt_answer = f"Pertanyaan: {generated_query}\n\nTeks: {doc_text}\n\nJawaban singkat:"
                    generated_answer = self.llm.generate(prompt_answer, max_tokens=150)
                    
                    # FULLY SUPPORTED: Answer directly from document
                    examples.append({
                        "instruction": f"""Query: {generated_query}

Answer: {generated_answer}

Documents:
{doc_text}

Is the answer fully supported, partially supported, or not supported by the documents?""",
                        "input": "",
                        "output": "[Fully Supported]",
                        "category": "support_verification"
                    })
                    
                    # PARTIALLY SUPPORTED: Add extra info to answer
                    if len(examples) < num_examples * 0.7:
                        partial_answer = generated_answer + " Selain itu, terdapat program tambahan yang tersedia."
                        examples.append({
                            "instruction": f"""Query: {generated_query}

Answer: {partial_answer}

Documents:
{doc_text}

Is the answer fully supported, partially supported, or not supported by the documents?""",
                            "input": "",
                            "output": "[Partially Supported]",
                            "category": "support_verification"
                        })
                    
                    # NO SUPPORT: Completely wrong answer
                    if len(examples) < num_examples * 0.9:
                        wrong_answer = "Informasi ini dapat dilihat di website resmi setiap hari Senin-Jumat pukul 08.00-16.00."
                        examples.append({
                            "instruction": f"""Query: {generated_query}

Answer: {wrong_answer}

Documents:
{doc_text}

Is the answer fully supported, partially supported, or not supported by the documents?""",
                            "input": "",
                            "output": "[No Support]",
                            "category": "support_verification"
                        })
                    
                except Exception as e:
                    app_logger.warning(f"Failed to generate support example with LLM: {e}")
                    continue
                
                if len(examples) >= num_examples:
                    break
        
        except Exception as e:
            app_logger.error(f"Failed to generate support examples: {e}")
        
        app_logger.info(f"Generated {len(examples)} support verification examples from real data")
        return examples
    
    def _generate_utility_examples(self, num_examples: int = 50) -> List[Dict]:
        """
        Generate utility evaluation examples using real documents
        """
        examples = []
        
        try:
            # Get documents from vector DB
            all_docs = self.vector_db.collection.get()
            
            if not all_docs.get("documents"):
                app_logger.warning("No documents in vector DB for utility examples")
                return examples
            
            documents = all_docs["documents"]
            metadatas = all_docs["metadatas"]
            
            # Sample documents
            indices = list(range(len(documents)))
            random.shuffle(indices)
            
            for idx in indices[:num_examples]:
                doc_text = documents[idx][:400]
                
                try:
                    # Generate query from document
                    prompt_query = f"Berdasarkan teks berikut, buat 1 pertanyaan:\n\n{doc_text}\n\nPertanyaan:"
                    query = self.llm.generate(prompt_query, max_tokens=80)
                    
                    # Generate different quality answers
                    utility_level = random.choice([1, 3, 5])
                    
                    if utility_level == 5:
                        # High utility: Detailed, comprehensive answer
                        prompt = f"Pertanyaan: {query}\n\nTeks: {doc_text}\n\nBerikan jawaban yang sangat lengkap dan detail:"
                        answer = self.llm.generate(prompt, max_tokens=200)
                        utility = "[Utility:5]"
                    
                    elif utility_level == 3:
                        # Medium utility: Basic answer
                        prompt = f"Pertanyaan: {query}\n\nTeks: {doc_text}\n\nJawaban singkat:"
                        answer = self.llm.generate(prompt, max_tokens=80)
                        utility = "[Utility:3]"
                    
                    else:  # utility_level == 1
                        # Low utility: Very short or unhelpful
                        answer = random.choice([
                            "Ya, benar.",
                            "Bisa dilihat di dokumen.",
                            "Informasi tersedia.",
                            "Silakan hubungi admin."
                        ])
                        utility = "[Utility:1]"
                    
                    examples.append({
                        "instruction": f"""Query: {query}

Answer: {answer}

Rate the utility of this answer for the query (1-5, where 5 is most useful):""",
                        "input": "",
                        "output": utility,
                        "category": "utility_evaluation"
                    })
                
                except Exception as e:
                    app_logger.warning(f"Failed to generate utility example: {e}")
                    continue
                
                if len(examples) >= num_examples:
                    break
        
        except Exception as e:
            app_logger.error(f"Failed to generate utility examples: {e}")
        
        app_logger.info(f"Generated {len(examples)} utility evaluation examples from real data")
        return examples
    
    def generate_full_dataset(
        self,
        num_retrieval: int = 200,
        num_relevance: int = 200,
        num_support: int = 100,
        num_utility: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        Generate complete training dataset
        
        Returns:
            Dict with train/validation/test splits
        """
        app_logger.info("Generating full Self-RAG training dataset...")
        
        # Generate all examples
        all_examples = []
        all_examples.extend(self._generate_retrieval_examples(num_retrieval))
        all_examples.extend(self._generate_relevance_examples(num_relevance))
        all_examples.extend(self._generate_support_examples(num_support))
        all_examples.extend(self._generate_utility_examples(num_utility))
        
        # Shuffle
        random.shuffle(all_examples)
        
        # Split: 80% train, 10% val, 10% test
        total = len(all_examples)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        
        dataset = {
            "train": all_examples[:train_end],
            "validation": all_examples[train_end:val_end],
            "test": all_examples[val_end:]
        }
        
        # Save to files
        for split, examples in dataset.items():
            output_file = self.output_dir / f"{split}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            app_logger.info(f"Saved {len(examples)} examples to {output_file}")
        
        # Save summary
        summary = {
            "total_examples": total,
            "train_size": len(dataset["train"]),
            "validation_size": len(dataset["validation"]),
            "test_size": len(dataset["test"]),
            "categories": {
                "retrieval_decision": num_retrieval,
                "relevance_check": num_relevance,
                "support_verification": num_support,
                "utility_evaluation": num_utility
            }
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        app_logger.info(f"Dataset generation complete! Summary saved to {summary_file}")
        
        return dataset


def main():
    """Main function to generate dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Self-RAG training dataset")
    parser.add_argument("--output-dir", type=str, default="./fine_tuning_data",
                       help="Output directory for dataset")
    parser.add_argument("--num-retrieval", type=int, default=200,
                       help="Number of retrieval decision examples")
    parser.add_argument("--num-relevance", type=int, default=200,
                       help="Number of relevance check examples")
    parser.add_argument("--num-support", type=int, default=100,
                       help="Number of support verification examples")
    parser.add_argument("--num-utility", type=int, default=100,
                       help="Number of utility evaluation examples")
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = SelfRAGDatasetGenerator(output_dir=args.output_dir)
    dataset = generator.generate_full_dataset(
        num_retrieval=args.num_retrieval,
        num_relevance=args.num_relevance,
        num_support=args.num_support,
        num_utility=args.num_utility
    )
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"\nFiles saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
