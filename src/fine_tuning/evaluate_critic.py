"""
Evaluation Script for Self-RAG Critic Model

Evaluates fine-tuned critic on test set
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import torch
from pathlib import Path
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import numpy as np

from src.config.logger import app_logger


class SelfRAGEvaluator:
    """Evaluate Self-RAG critic model"""
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str = "./fine_tuning_data/test.jsonl"
    ):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to fine-tuned model
            test_data_path: Path to test dataset
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        app_logger.info(f"Initializing evaluator on {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load test data
        self._load_test_data()
    
    def _load_model(self):
        """Load fine-tuned model"""
        app_logger.info(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.model.eval()
        app_logger.info("Model loaded successfully")
    
    def _load_test_data(self):
        """Load test dataset"""
        app_logger.info(f"Loading test data from: {self.test_data_path}")
        
        self.test_examples = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.test_examples.append(json.loads(line))
        
        app_logger.info(f"Loaded {len(self.test_examples)} test examples")
    
    def predict(self, instruction: str, input_text: str = "") -> str:
        """
        Make prediction with model
        
        Args:
            instruction: Task instruction
            input_text: Optional input
            
        Returns:
            Model prediction
        """
        # Format prompt
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}\n\nOutput:"
        else:
            prompt = f"{instruction}\n\nOutput:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the output part
        if "Output:" in generated_text:
            prediction = generated_text.split("Output:")[-1].strip()
        else:
            prediction = generated_text.strip()
        
        return prediction
    
    def evaluate_category(self, category: str) -> Dict:
        """
        Evaluate specific category
        
        Args:
            category: Category to evaluate
            
        Returns:
            Metrics dictionary
        """
        app_logger.info(f"Evaluating category: {category}")
        
        # Filter examples by category
        category_examples = [
            ex for ex in self.test_examples 
            if ex.get("category") == category
        ]
        
        if not category_examples:
            app_logger.warning(f"No examples found for category: {category}")
            return {}
        
        correct = 0
        total = len(category_examples)
        predictions = []
        ground_truths = []
        
        # Evaluate each example
        for example in tqdm(category_examples, desc=f"Evaluating {category}"):
            instruction = example["instruction"]
            input_text = example.get("input", "")
            expected_output = example["output"]
            
            # Predict
            prediction = self.predict(instruction, input_text)
            predictions.append(prediction)
            ground_truths.append(expected_output)
            
            # Check if correct
            if expected_output.lower() in prediction.lower():
                correct += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            "category": category,
            "total_examples": total,
            "correct": correct,
            "accuracy": accuracy,
            "predictions": predictions[:10],  # Sample predictions
            "ground_truths": ground_truths[:10]
        }
        
        app_logger.info(f"{category} - Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return metrics
    
    def evaluate_all(self) -> Dict:
        """
        Evaluate all categories
        
        Returns:
            Complete evaluation results
        """
        app_logger.info("Starting full evaluation...")
        
        # Get all categories
        categories = set(ex.get("category") for ex in self.test_examples)
        
        results = {
            "model_path": self.model_path,
            "total_examples": len(self.test_examples),
            "categories": {}
        }
        
        # Evaluate each category
        for category in sorted(categories):
            if category:
                metrics = self.evaluate_category(category)
                results["categories"][category] = metrics
        
        # Calculate overall accuracy
        total_correct = sum(
            m.get("correct", 0) 
            for m in results["categories"].values()
        )
        total_examples = sum(
            m.get("total_examples", 0) 
            for m in results["categories"].values()
        )
        
        results["overall_accuracy"] = total_correct / total_examples if total_examples > 0 else 0
        
        # Save results
        output_file = Path(self.model_path) / "evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        app_logger.info(f"Evaluation complete! Results saved to {output_file}")
        app_logger.info(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("Self-RAG Critic Model Evaluation Results")
        print("="*70)
        print(f"Model: {results['model_path']}")
        print(f"Total Examples: {results['total_examples']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print("\n" + "-"*70)
        print("Per-Category Results:")
        print("-"*70)
        
        for category, metrics in sorted(results["categories"].items()):
            accuracy = metrics.get("accuracy", 0)
            correct = metrics.get("correct", 0)
            total = metrics.get("total_examples", 0)
            
            print(f"\n{category}:")
            print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
            
            # Show sample predictions
            if metrics.get("predictions"):
                print(f"\n  Sample Predictions:")
                for i, (pred, gt) in enumerate(zip(
                    metrics["predictions"][:3],
                    metrics["ground_truths"][:3]
                ), 1):
                    print(f"    {i}. Expected: {gt}")
                    print(f"       Predicted: {pred}")
                    print()
        
        print("="*70 + "\n")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Self-RAG critic model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--test-data", type=str, default="./fine_tuning_data/test.jsonl",
                       help="Path to test dataset")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = SelfRAGEvaluator(
        model_path=args.model_path,
        test_data_path=args.test_data
    )
    
    results = evaluator.evaluate_all()
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
