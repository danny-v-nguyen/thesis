import os
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. SETUP PATHS & LOAD SAVED PROBE
# ==========================================
# Point to your Layer 12 probe pkl file
PROBE_PATH = "./probe_cache_multi/probe_layer_12.pkl" # Adjust path if using multi-layer cache

if not os.path.exists(PROBE_PATH):
    raise FileNotFoundError(
        f"Could not find the trained probe at {PROBE_PATH}. "
        "Please ensure your training script completed and saved the .pkl file."
    )

with open(PROBE_PATH, "rb") as f:
    probe = pickle.load(f)

# The reverse mapping of your training labels
CLASS_NAMES = {0: "fact", 1: "implication", 2: "negation"}

# ==========================================
# 2. DEFINE ADVERSARIAL TEST CASESS
# ==========================================
adversarial_examples = [
    # --- Category A: Syntactic distractors (Have "if" but are NOT logical implications) ---
    {"text": "John wondered if the library was still open.", "expected": "fact", "note": "Embedded question, not logical rule"},
    {"text": "I asked her if she wanted some tea.", "expected": "fact", "note": "Indirect question"},
    {"text": "If only I had remembered my keys!", "expected": "fact", "note": "Subjunctive wish, not conditional"},
    {"text": "She acts as if nothing happened.", "expected": "fact", "note": "Comparison structure"},

    # --- Category B: Standard Implications (To prove the probe still functions normally) ---
    {"text": "If a person is green, then they are smart.", "expected": "implication", "note": "Standard LogicNLI template"},
    {"text": "If it rains, then the ground gets wet.", "expected": "implication", "note": "Standard logical implication"},
    
    # --- Category C: Facts/Negations ---
    {"text": "The apple is red and sweet.", "expected": "fact", "note": "Simple assertion"},
    {"text": "The cat is not on the kitchen table.", "expected": "negation", "note": "Negated fact"}
]

sentences = [ex["text"] for ex in adversarial_examples]

# ==========================================
# 3. INITIALIZE MODEL & EXTRACTION HOOK (Layer 12)
# ==========================================
print("Loading Llama 3.2-3B...")
model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")

extracted_activations = {}
def residual_stream_hook(module, input, output):
    if isinstance(output, (tuple, list)):
        extracted_activations["hidden_states"] = output[0]
    else:
        extracted_activations["hidden_states"] = output

# Hook into Layer 12
target_layer = model.model.layers[12]
hook_handle = target_layer.register_forward_hook(residual_stream_hook)

# ==========================================
# 4. RUN INFERENCE & FEATURE EXTRACTION
# ==========================================
print("\nExtracting features from adversarial inputs...")
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    model(**inputs)

layer_hidden_states = extracted_activations["hidden_states"]
attention_mask = inputs["attention_mask"]

# Masked Mean Pooling
input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_states.size()).float()
sum_embeddings = torch.sum(layer_hidden_states * input_mask_expanded, dim=1)
sum_mask = input_mask_expanded.sum(dim=1)
sum_mask = torch.clamp(sum_mask, min=1e-9)
mean_pooled_vectors = (sum_embeddings / sum_mask).detach().cpu().numpy()

# Clean up model hooks
hook_handle.remove()

# ==========================================
# 5. TEST THE PROBE AND PRINT FINDINGS
# ==========================================
print("\n=== ADVERSARIAL TEST RESULTS ===\n")
# Calculate probability distributions to see confidence
probs = probe.predict_proba(mean_pooled_vectors)
predictions = probe.predict(mean_pooled_vectors)

print(f"{'Test Sentence':<45} | {'Expected':<12} | {'Predicted Class':<15} | {'Match?':<6} | {'Confidence (Impl.)'}")
print("-" * 105)

for i, example in enumerate(adversarial_examples):
    pred_idx = predictions[i]
    predicted_label = CLASS_NAMES[pred_idx]
    
    # Check probability score assigned to the "implication" class (index 1)
    impl_prob = probs[i][1] * 100 
    
    is_match = "YES" if predicted_label == example["expected"] else "NO (FAIL)"
    
    # We flag suspicious false-positives
    status_indicator = "⚠️ SHIFT" if (example["expected"] == "fact" and predicted_label == "implication") else ""
    
    print(f"{example['text']:<45} | {example['expected']:<12} | {predicted_label:<15} | {is_match:<6} | {impl_prob:5.1f}% {status_indicator}")

print("\n=================================")
print("KEY:")
print("⚠️ SHIFT = Sentence was expected to be a Fact, but the probe classified it as an Implication.")