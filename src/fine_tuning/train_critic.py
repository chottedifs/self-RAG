"""
Fine-tuning Script for Self-RAG Critic Model

Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import torch
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import wandb
from transformers.trainer_callback import ProgressCallback

from src.config.logger import app_logger


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    # Model settings
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Lightweight model for CPU training
    model_max_length: int = 2048
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Training settings
    output_dir: str = "./models/self_rag_critic"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Reduced to 1 for CPU training (prevent hanging)
    per_device_eval_batch_size: int = 1  # Reduced to 1 for CPU
    gradient_accumulation_steps: int = 8  # Increased to compensate for smaller batch size
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    # Optimization
    fp16: bool = True
    optim: str = "adamw_torch"
    
    # Data settings
    train_data_path: str = "./fine_tuning_data/train.jsonl"
    val_data_path: str = "./fine_tuning_data/validation.jsonl"
    
    # Wandb settings
    use_wandb: bool = False
    wandb_project: str = "self-rag-critic"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class SelfRAGFineTuner:
    """Fine-tune LLM for Self-RAG critic tasks"""
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize fine-tuner
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        app_logger.info(f"Initializing Self-RAG fine-tuner on {self.device}")
        app_logger.info(f"Base model: {config.base_model}")
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=config.__dict__)
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        app_logger.info(f"Loading model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=self.config.model_max_length,
            padding_side="right",
            use_fast=False
        )
        
        # Add special tokens for Self-RAG
        special_tokens = {
            "additional_special_tokens": [
                "[Retrieve]", "[No Retrieve]",
                "[Relevant]", "[Irrelevant]",
                "[Fully Supported]", "[Partially Supported]", "[No Support]",
                "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        try:
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    load_in_8bit=True,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    local_files_only=False
                )
            else:
                # For CPU: explicit single device loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    local_files_only=False
                )
                # Explicitly move to CPU
                self.model = self.model.to("cpu")
        except Exception as e:
            app_logger.error(f"Failed to load model: {str(e)}")
            app_logger.info("Retrying with different settings...")
            # Fallback: try loading without device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
        
        # Resize token embeddings for new special tokens
        # Set mean_resizing=False to disable warning about multivariate normal initialization
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        
        app_logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        app_logger.info("Setting up LoRA...")
        
        # Prepare model for k-bit training
        if self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Ensure model is on correct device after LoRA wrapping
        if self.device == "cpu":
            # For CPU: force single device and disable any device parallelism
            self.model = self.model.to("cpu")
            # Disable gradient checkpointing which can cause issues on CPU
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
        
        app_logger.info("LoRA setup complete")
    
    def load_datasets(self):
        """Load and preprocess training datasets"""
        app_logger.info("Loading datasets...")
        
        # Load train and validation datasets
        data_files = {
            "train": self.config.train_data_path,
            "validation": self.config.val_data_path
        }
        
        self.dataset = load_dataset("json", data_files=data_files)
        
        # Preprocess function
        def preprocess_function(examples):
            """Format examples for training"""
            prompts = []
            for instruction, input_text, output in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            ):
                if input_text:
                    text = f"{instruction}\n\nInput: {input_text}\n\nOutput: {output}"
                else:
                    text = f"{instruction}\n\nOutput: {output}"
                
                prompts.append(text)
            
            # Tokenize
            model_inputs = self.tokenizer(
                prompts,
                max_length=self.config.model_max_length,
                truncation=True,
                padding="max_length"
            )
            
            # Set labels (same as input_ids for causal LM)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        # Apply preprocessing
        self.tokenized_dataset = self.dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
        app_logger.info(f"Train examples: {len(self.tokenized_dataset['train'])}")
        app_logger.info(f"Validation examples: {len(self.tokenized_dataset['validation'])}")
    
    def train(self):
        """Train the model"""
        app_logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",  # Updated from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            fp16=self.config.fp16 and self.device == "cuda",
            optim=self.config.optim,
            report_to="wandb" if self.config.use_wandb else "none",
            logging_dir=f"{self.config.output_dir}/logs",
            # Fix warnings for CPU/Intel Iris training
            use_cpu=True if self.device == "cpu" else False,  # Use use_cpu instead of deprecated no_cuda
            dataloader_pin_memory=False,  # Disable pin_memory for CPU/Intel Iris
            ddp_find_unused_parameters=False if self.device == "cpu" else None,  # Avoid DDP issues on CPU
            local_rank=-1,  # Disable distributed training (no multi-device)
            dataloader_num_workers=0,  # Single worker for CPU to avoid multiprocessing overhead
            # Additional CPU optimizations to prevent hanging
            gradient_checkpointing=False,  # Disable gradient checkpointing on CPU
            max_grad_norm=1.0,  # Clip gradients to prevent instability
            remove_unused_columns=True,  # Clean up unused data to save memory
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=data_collator,
            callbacks=[ProgressCallback()]  # Add progress callback for visibility
        )
        
        app_logger.info(f"Starting training with {len(self.tokenized_dataset['train'])} examples...")
        app_logger.info(f"Batch size: {self.config.per_device_train_batch_size}, Gradient accumulation: {self.config.gradient_accumulation_steps}")
        app_logger.info(f"Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        app_logger.info("This may take several hours on CPU. Please be patient...")
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        app_logger.info(f"Training complete! Model saved to {self.config.output_dir}")
    
    def run_full_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        try:
            self.load_model_and_tokenizer()
            self.setup_lora()
            self.load_datasets()
            self.train()
            
            app_logger.info("Fine-tuning pipeline completed successfully!")
            
        except Exception as e:
            app_logger.error(f"Fine-tuning failed: {e}")
            raise


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Self-RAG critic model")
    parser.add_argument("--base-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Base model to fine-tune (default: TinyLlama for CPU efficiency)")
    parser.add_argument("--output-dir", type=str, default="./models/self_rag_critic_tiny",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--train-data", type=str, default="./fine_tuning_data/train.jsonl",
                       help="Path to training data")
    parser.add_argument("--val-data", type=str, default="./fine_tuning_data/validation.jsonl",
                       help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = FineTuningConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )
    
    # Run fine-tuning
    print("\n" + "="*60)
    print("Self-RAG Critic Model Fine-Tuning")
    print("="*60)
    print(f"Base Model: {config.base_model}")
    print(f"Output Dir: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch Size: {config.per_device_train_batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print("="*60 + "\n")
    
    fine_tuner = SelfRAGFineTuner(config)
    fine_tuner.run_full_pipeline()


if __name__ == "__main__":
    main()
