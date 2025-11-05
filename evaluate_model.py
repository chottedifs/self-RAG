"""
Evaluation script for trained Self-RAG Critic Model
Evaluates on 4 tasks: retrieval decision, relevance check, support verification, utility evaluation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import json
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class CriticEvaluator:
    """
    Evaluator for Self-RAG Critic Model
    """
    
    def __init__(
        self,
        model_path: str,
        base_model: str = "mistral:latest",
        test_file: str = "./fine_tuning_data/test.jsonl",
        device: str = None
    ):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained LoRA model
            base_model: Base model name
            test_file: Path to test data
            device: Device to use (auto-detect if None)
        """
        # Convert to absolute path to avoid HuggingFace validation errors
        import os
        self.model_path = os.path.abspath(model_path)
        self.base_model = base_model
        self.test_file = os.path.abspath(test_file) if not os.path.isabs(test_file) else test_file
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Loading model from {model_path}...")
        self._load_model()
        
        print(f"📂 Loading test data from {test_file}...")
        self.test_data = self._load_test_data()
        
        # Task mapping
        self.task_labels = {
            "retrieval_decision": ["Yes", "No"],
            "relevance_check": ["Relevant", "Irrelevant"],
            "support_verification": ["Fully Supported", "Partially Supported", "No Support"],
            "utility_evaluation": ["1", "2", "3", "4", "5"]
        }
        
    def _load_model(self):
        """Load trained LoRA model"""
        # Load tokenizer (with special tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Resize embeddings to match tokenizer
        if base.config.vocab_size != len(self.tokenizer):
            base.resize_token_embeddings(len(self.tokenizer))
            print(f"✓ Embeddings resized: {base.config.vocab_size} → {len(self.tokenizer)}")
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base, self.model_path)
        self.model.eval()  # Set to eval mode
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        print(f"✓ Model loaded on {self.device}")
        
    def _load_test_data(self) -> Dict[str, List[Dict]]:
        """Load test data grouped by task"""
        data_by_task = {
            "retrieval_decision": [],
            "relevance_check": [],
            "support_verification": [],
            "utility_evaluation": []
        }
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Try both 'task' and 'category' fields
                task = item.get('task') or item.get('category')
                if task in data_by_task:
                    data_by_task[task].append(item)
        
        print(f"✓ Test data loaded:")
        for task, items in data_by_task.items():
            print(f"  {task}: {len(items)} examples")
        
        return data_by_task
    
    def _format_prompt(self, example: Dict) -> str:
        """Format example into prompt - match training format exactly"""
        instruction = example['instruction']
        input_text = example.get('input', '')
        
        # Match training format exactly (without trailing space after "Output:")
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}\n\nOutput:"
        else:
            prompt = f"{instruction}\n\nOutput:"
        
        return prompt
    
    def _predict(self, prompt: str) -> str:
        """Get model prediction"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # Increased from 20
                min_new_tokens=1,   # Force at least 1 token
                do_sample=True,     # Enable sampling
                temperature=0.7,    # Increased from 0.1 for more diversity
                top_p=0.9,          # Nucleus sampling
                repetition_penalty=1.1,  # Prevent repetition
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1  # Greedy decode
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only generated part (after prompt)
        # Try to find "Output:" and take text after it
        if "Output:" in prediction:
            parts = prediction.split("Output:")
            if len(parts) > 1:
                generated = parts[-1].strip()
            else:
                generated = prediction[len(prompt):].strip()
        else:
            generated = prediction[len(prompt):].strip()
        
        return generated
    
    def _extract_label(self, prediction: str, task: str) -> str:
        """Extract label from model prediction"""
        prediction_lower = prediction.strip().lower()
        
        # First check for exact special token match (with brackets)
        for label in self.task_labels[task]:
            if label.lower() in prediction_lower or label.replace(" ", "").lower() in prediction_lower:
                return label
        
        # Task-specific extraction rules (natural language outputs)
        if task == "retrieval_decision":
            # Map natural language to expected labels
            if any(word in prediction_lower for word in ["yes", "retrieve", "need", "necessary", "should"]):
                return "Yes"
            elif any(word in prediction_lower for word in ["no", "don't", "not", "unnecessary", "shouldn't"]):
                return "No"
            # Fallback: first word
            first_word = prediction_lower.split()[0] if prediction_lower else "no"
            return "Yes" if first_word == "yes" else "No"
        
        elif task == "relevance_check":
            if "irrelevant" in prediction_lower or "not relevant" in prediction_lower or "unrelated" in prediction_lower:
                return "Irrelevant"
            elif "relevant" in prediction_lower or "related" in prediction_lower:
                return "Relevant"
            # Fallback
            first_word = prediction_lower.split()[0] if prediction_lower else "irrelevant"
            return "Relevant" if first_word in ["yes", "relevant"] else "Irrelevant"
        
        elif task == "support_verification":
            if "fully supported" in prediction_lower or "full support" in prediction_lower or "completely" in prediction_lower:
                return "Fully Supported"
            elif "no support" in prediction_lower or "not supported" in prediction_lower or "unsupported" in prediction_lower:
                return "No Support"
            elif "partially" in prediction_lower or "partial" in prediction_lower or "somewhat" in prediction_lower:
                return "Partially Supported"
            # Fallback: analyze sentiment
            if "no" in prediction_lower[:20]:  # Check first 20 chars
                return "No Support"
            elif "partial" in prediction_lower or "some" in prediction_lower:
                return "Partially Supported"
            else:
                return "Fully Supported"
        
        elif task == "utility_evaluation":
            # Extract number 1-5
            import re
            numbers = re.findall(r'\b[1-5]\b', prediction)
            if numbers:
                return numbers[0]
            # Fallback: try to infer from text
            if any(word in prediction_lower for word in ["excellent", "perfect", "very useful"]):
                return "5"
            elif any(word in prediction_lower for word in ["good", "useful"]):
                return "3"
            elif any(word in prediction_lower for word in ["poor", "useless", "not helpful"]):
                return "1"
            return "3"  # Default middle
        
        # Last resort: return first token or Unknown
        return prediction.split()[0] if prediction else "Unknown"
    
    def evaluate_task(self, task: str) -> Dict:
        """
        Evaluate model on specific task
        
        Args:
            task: Task name (retrieval_decision, relevance_check, etc.)
            
        Returns:
            Dict with metrics
        """
        examples = self.test_data[task]
        
        if not examples:
            print(f"⚠️  No test data for task: {task}")
            return {}
        
        print(f"\n🔍 Evaluating {task} ({len(examples)} examples)...")
        
        predictions = []
        ground_truths = []
        
        for i, example in enumerate(tqdm(examples, desc=f"Evaluating {task}"), 1):
            try:
                # Format prompt
                prompt = self._format_prompt(example)
                
                # Get prediction
                pred_text = self._predict(prompt)
                pred_label = self._extract_label(pred_text, task)
                
                # Ground truth - remove brackets
                true_label = example['output'].strip().replace("[", "").replace("]", "")
                
                predictions.append(pred_label)
                ground_truths.append(true_label)
                
                # Debug: print first few
                if i <= 2:
                    print(f"\n  Example {i}: Expected '{true_label}' | Predicted '{pred_label}' | Raw '{pred_text[:80]}...'")
                    
            except Exception as e:
                print(f"\n❌ Error on example {i}: {e}")
                predictions.append("Unknown")
                ground_truths.append(example['output'].strip().replace("[", "").replace("]", ""))
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            predictions,
            ground_truths,
            self.task_labels[task]
        )
        
        # Print results
        print(f"\n📊 Results for {task}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Macro Precision: {metrics['precision']:.2%}")
        print(f"  Macro Recall: {metrics['recall']:.2%}")
        print(f"  Macro F1: {metrics['f1']:.2%}")
        
        # Detailed classification report
        print("\n  Classification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str],
        labels: List[str]
    ) -> Dict:
        """Calculate evaluation metrics"""
        
        # Accuracy
        accuracy = accuracy_score(ground_truths, predictions)
        
        # Precision, Recall, F1 (macro average)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truths,
            predictions,
            labels=labels,
            average='macro',
            zero_division=0
        )
        
        # Detailed classification report
        report = classification_report(
            ground_truths,
            predictions,
            labels=labels,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
    
    def evaluate_all_tasks(self) -> Dict[str, Dict]:
        """Evaluate model on all tasks"""
        print("\n" + "="*60)
        print("🚀 STARTING EVALUATION ON ALL TASKS")
        print("="*60)
        
        results = {}
        
        for task in self.task_labels.keys():
            task_results = self.evaluate_task(task)
            if task_results:  # Only add non-empty results
                results[task] = task_results
        
        # Calculate average metrics (only for tasks with data)
        if results:
            avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
            avg_f1 = sum(r['f1'] for r in results.values()) / len(results)
            
            print("\n" + "="*60)
            print("📊 OVERALL RESULTS")
            print("="*60)
            print(f"Tasks evaluated: {len(results)}/{len(self.task_labels)}")
            print(f"Average Accuracy: {avg_accuracy:.2%}")
            print(f"Average F1-Score: {avg_f1:.2%}")
        else:
            print("\n⚠️  WARNING: No tasks had test data to evaluate!")
        
        return results
    
    def save_results(self, output_file: str):
        """Save evaluation results to file"""
        results = self.evaluate_all_tasks()
        
        if not results:
            print("\n⚠️  No results to save (no test data found)")
            return
        
        # Prepare serializable results (remove numpy arrays, etc.)
        serializable_results = {}
        for task, metrics in results.items():
            serializable_results[task] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'classification_report': metrics['classification_report']
            }
        
        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to {output_file}")


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Self-RAG Critic Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/self_rag_critic",
        help="Path to trained model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./fine_tuning_data/test.jsonl",
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./logs/evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CriticEvaluator(
        model_path=args.model_path,
        base_model=args.base_model,
        test_file=args.test_file
    )
    
    # Run evaluation and save results
    evaluator.save_results(args.output)


if __name__ == "__main__":
    main()