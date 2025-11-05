"""
Test script untuk model yang sudah di-training
"""
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to trained model
MODEL_PATH = "./models/self_rag_critic"
BASE_MODEL = "mistral:latest"

print("🔄 Loading trained model...")

# Load tokenizer (from trained model, which has special tokens)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"✓ Tokenizer loaded (vocab: {len(tokenizer)})")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"✓ Base model loaded (original vocab: {base_model.config.vocab_size})")

# ⭐ KEY FIX: Resize base model embeddings to match trained model
base_model.resize_token_embeddings(len(tokenizer))
print(f"✓ Base model embeddings resized to {len(tokenizer)}")

# Now load LoRA adapters (should work!)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
print(f"✓ LoRA adapters loaded")

# Test inference
print("\n" + "="*60)
print("🧪 TEST INFERENCE")
print("="*60)

test_cases = [
    {
        "instruction": "Query: Berapa biaya SKS semester ini?\n\nApakah query ini memerlukan pencarian dokumen?",
        "expected": "[Retrieve] atau Yes"
    },
    {
        "instruction": "Query: Halo\nDocument: (tidak ada)\n\nApakah dokumen ini relevan dengan query?",
        "expected": "[Irrelevant] atau No"
    },
    {
        "instruction": "Query: Berapa SKS minimal?\nAnswer: Minimal 12 SKS per semester.\nDocument: Mahasiswa wajib mengambil minimal 12 SKS.\n\nApakah jawaban didukung oleh dokumen?",
        "expected": "[Fully Supported]"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Input: {test['instruction'][:100]}...")
    print(f"Expected: {test['expected']}")
    
    inputs = tokenizer(test['instruction'], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after input)
    generated = prediction[len(test['instruction']):].strip()
    print(f"Generated: {generated}")
    print()

print("="*60)
print("✓ Model testing complete!")