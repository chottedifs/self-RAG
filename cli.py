#!/usr/bin/env python3
"""
RAG Chatbot CLI - Unified Command Line Interface

Commands:
  prepare-data    Process PDFs and build vector database
  train-critic    Fine-tune Self-RAG critic model
  eval-critic     Evaluate fine-tuned critic model
  run-ui          Launch Streamlit web interface
  run-api         Launch FastAPI server
  generate-dataset Generate Self-RAG training dataset
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import subprocess
from pathlib import Path

from src.config.logger import app_logger


def prepare_data(args):
    """Prepare data: extract PDFs, chunk, embed, store to ChromaDB"""
    print("\n" + "="*70)
    print("📄 DATA PREPARATION")
    print("="*70)
    
    cmd = [sys.executable, "src/data_processing/prepare_data.py"]
    
    if args.clear:
        cmd.append("--clear")
        print("⚠️  Clear existing data: YES")
    
    if args.file:
        cmd.extend(["--file", args.file])
        print(f"📎 Single file: {args.file}")
    else:
        print(f"📁 Processing directory: data_resources/")
    
    print("="*70 + "\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Data preparation completed successfully!")
    else:
        print("\n❌ Data preparation failed!")
        sys.exit(1)


def generate_dataset(args):
    """Generate Self-RAG training dataset"""
    print("\n" + "="*70)
    print("🎯 GENERATE SELF-RAG DATASET")
    print("="*70)
    
    cmd = [
        sys.executable, "src/fine_tuning/prepare_dataset.py",
        "--output-dir", args.output_dir,
        "--num-retrieval", str(args.num_retrieval),
        "--num-relevance", str(args.num_relevance),
        "--num-support", str(args.num_support),
        "--num-utility", str(args.num_utility)
    ]
    
    print(f"📊 Output: {args.output_dir}")
    print(f"📝 Retrieval examples: {args.num_retrieval}")
    print(f"📝 Relevance examples: {args.num_relevance}")
    print(f"📝 Support examples: {args.num_support}")
    print(f"📝 Utility examples: {args.num_utility}")
    print("="*70 + "\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Dataset generation completed!")
    else:
        print("\n❌ Dataset generation failed!")
        sys.exit(1)


def train_critic(args):
    """Fine-tune Self-RAG critic model"""
    print("\n" + "="*70)
    print("🎓 FINE-TUNE SELF-RAG CRITIC")
    print("="*70)
    
    cmd = [
        sys.executable, "src/fine_tuning/train_critic.py",
        "--base-model", args.base_model,
        "--output-dir", args.output_dir,
        "--train-data", args.train_data,
        "--val-data", args.val_data,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate)
    ]
    
    if args.use_wandb:
        cmd.append("--use-wandb")
    
    print(f"🤖 Base model: {args.base_model}")
    print(f"💾 Output: {args.output_dir}")
    print(f"📚 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print("="*70 + "\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Fine-tuning completed!")
    else:
        print("\n❌ Fine-tuning failed!")
        sys.exit(1)


def eval_critic(args):
    """Evaluate fine-tuned critic model"""
    print("\n" + "="*70)
    print("📊 EVALUATE CRITIC MODEL")
    print("="*70)
    
    cmd = [
        sys.executable, "src/fine_tuning/evaluate_critic.py",
        "--model-path", args.model_path,
        "--test-data", args.test_data
    ]
    
    print(f"🤖 Model: {args.model_path}")
    print(f"📝 Test data: {args.test_data}")
    print("="*70 + "\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Evaluation completed!")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


def run_ui(args):
    """Launch Streamlit UI"""
    print("\n" + "="*70)
    print("🚀 LAUNCHING STREAMLIT UI")
    print("="*70)
    
    ui_file = "src/ui/app.py"
    
    if args.self_rag and Path("src/ui/app_self_rag.py").exists():
        ui_file = "src/ui/app_self_rag.py"
        print("🧠 Mode: Self-RAG")
    else:
        print("🤖 Mode: Standard RAG")
    
    print(f"🌐 URL: http://localhost:{args.port}")
    print("="*70 + "\n")
    
    cmd = [
        "streamlit", "run", ui_file,
        "--server.port", str(args.port),
        "--server.headless", "true"
    ]
    
    subprocess.run(cmd)


def run_api(args):
    """Launch FastAPI server"""
    print("\n" + "="*70)
    print("🚀 LAUNCHING FASTAPI SERVER")
    print("="*70)
    
    # Check if api file exists
    api_files = ["src/api/main.py", "api/main.py", "run_api.py"]
    api_file = None
    
    for f in api_files:
        if Path(f).exists():
            api_file = f
            break
    
    if not api_file:
        print("❌ API file not found! Create src/api/main.py first.")
        sys.exit(1)
    
    print(f"📡 API: http://localhost:{args.port}")
    print(f"📚 Docs: http://localhost:{args.port}/docs")
    print("="*70 + "\n")
    
    cmd = [
        "uvicorn",
        api_file.replace("/", ".").replace("\\", ".").replace(".py", "") + ":app",
        "--host", args.host,
        "--port", str(args.port),
        "--reload"
    ]
    
    subprocess.run(cmd)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot CLI - Unified Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data
  python cli.py prepare-data
  python cli.py prepare-data --clear
  python cli.py prepare-data --file path/to/file.pdf
  
  # Generate Self-RAG dataset
  python cli.py generate-dataset
  
  # Train critic model
  python cli.py train-critic --epochs 5
  
  # Evaluate model
  python cli.py eval-critic --model-path ./models/self_rag_critic
  
  # Run UI
  python cli.py run-ui
  python cli.py run-ui --self-rag --port 8502
  
  # Run API
  python cli.py run-api --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ===== PREPARE-DATA =====
    parser_prepare = subparsers.add_parser(
        'prepare-data',
        help='Process PDFs and build vector database'
    )
    parser_prepare.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before processing'
    )
    parser_prepare.add_argument(
        '--file',
        type=str,
        help='Process a single PDF file'
    )
    parser_prepare.set_defaults(func=prepare_data)
    
    # ===== GENERATE-DATASET =====
    parser_gen = subparsers.add_parser(
        'generate-dataset',
        help='Generate Self-RAG training dataset'
    )
    parser_gen.add_argument(
        '--output-dir',
        type=str,
        default='./fine_tuning_data',
        help='Output directory for dataset'
    )
    parser_gen.add_argument(
        '--num-retrieval',
        type=int,
        default=200,
        help='Number of retrieval decision examples'
    )
    parser_gen.add_argument(
        '--num-relevance',
        type=int,
        default=200,
        help='Number of relevance check examples'
    )
    parser_gen.add_argument(
        '--num-support',
        type=int,
        default=100,
        help='Number of support verification examples'
    )
    parser_gen.add_argument(
        '--num-utility',
        type=int,
        default=100,
        help='Number of utility evaluation examples'
    )
    parser_gen.set_defaults(func=generate_dataset)
    
    # ===== TRAIN-CRITIC =====
    parser_train = subparsers.add_parser(
        'train-critic',
        help='Fine-tune Self-RAG critic model'
    )
    parser_train.add_argument(
        '--base-model',
        type=str,
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='Base model to fine-tune (default: TinyLlama for CPU efficiency)'
    )
    parser_train.add_argument(
        '--output-dir',
        type=str,
        default='./models/self_rag_critic_tiny',
        help='Output directory for fine-tuned model (TinyLlama version)'
    )
    parser_train.add_argument(
        '--train-data',
        type=str,
        default='./fine_tuning_data/train.jsonl',
        help='Path to training data'
    )
    parser_train.add_argument(
        '--val-data',
        type=str,
        default='./fine_tuning_data/validation.jsonl',
        help='Path to validation data'
    )
    parser_train.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser_train.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    parser_train.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    parser_train.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser_train.set_defaults(func=train_critic)
    
    # ===== EVAL-CRITIC =====
    parser_eval = subparsers.add_parser(
        'eval-critic',
        help='Evaluate fine-tuned critic model'
    )
    parser_eval.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to fine-tuned model'
    )
    parser_eval.add_argument(
        '--test-data',
        type=str,
        default='./fine_tuning_data/test.jsonl',
        help='Path to test dataset'
    )
    parser_eval.set_defaults(func=eval_critic)
    
    # ===== RUN-UI =====
    parser_ui = subparsers.add_parser(
        'run-ui',
        help='Launch Streamlit web interface'
    )
    parser_ui.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run UI on'
    )
    parser_ui.add_argument(
        '--self-rag',
        action='store_true',
        help='Use Self-RAG interface (if available)'
    )
    parser_ui.set_defaults(func=run_ui)
    
    # ===== RUN-API =====
    parser_api = subparsers.add_parser(
        'run-api',
        help='Launch FastAPI server'
    )
    parser_api.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to'
    )
    parser_api.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run API on'
    )
    parser_api.set_defaults(func=run_api)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
