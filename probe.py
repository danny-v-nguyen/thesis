import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ==========================================
# 1. INITIALIZE INFRASTRUCTURE
# ==========================================
model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Crucial for batching: Set a padding token and specify padding side
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Global dictionary to catch activations
extracted_activations = {}
def residual_stream_hook(module, input, output):
    # Keep the tensor on GPU or clone safely during batching
    extracted_activations["hidden_states"] = output[0]

target_layer = model.model.layers[12]
hook_handle = target_layer.register_forward_hook(residual_stream_hook)

LABEL_MAP = {"fact": 0, "implication": 1, "negation": 2}

# ==========================================
# 2. PYTORCH BATCHING DATASET
# ==========================================
class LogicNLIDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).dropna(subset=["Sentence"])
        # Filter rows to make sure labels match our expectations
        df = df[df["Label"].isin(LABEL_MAP.keys())].reset_index(drop=True)
        
        self.sentences = df["Sentence"].tolist()
        self.labels = [LABEL_MAP[l] for l in df["Label"]]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

# ==========================================
# 3. BATTERED EXTRACTION ENGINE
# ==========================================
def extract_features_batched(csv_path, batch_size=32, split_name="Split"):
    dataset = LogicNLIDataset(csv_path)
    # DataLoader handles parallel grouping and shuffling
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    X_list = []
    Y_list = []
    
    progress_description = f"Extracting {split_name:<5}"
    
    for batch_sentences, batch_labels in tqdm(dataloader, desc=progress_description, unit="batch"):
        # Tokenize the entire batch simultaneously with padding
        inputs = tokenizer(
            list(batch_sentences), 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        # Push tensors to the GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model(**inputs)
            
        # layer_hidden_states shape: [batch_size, max_seq_len_in_batch, 3072]
        layer_hidden_states = extracted_activations["hidden_states"]
        attention_mask = inputs["attention_mask"] # Shape: [batch_size, max_seq_len_in_batch]
        
        # --- MASKED MEAN-POOLING ---
        # Crucial: When batching, we must ignore padding tokens when calculating the mean!
        # Expand attention mask to match hidden states dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_states.size()).float()
        
        # Multiply hidden states by the mask (zeros out the padding tokens vectors)
        sum_embeddings = torch.sum(layer_hidden_states * input_mask_expanded, dim=1)
        
        # Count actual non-padding tokens per sentence
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # Prevent division by zero
        
        # Complete masked mean pooling
        mean_pooled_batch = (sum_embeddings / sum_mask).detach().cpu().numpy()
        
        X_list.append(mean_pooled_batch)
        Y_list.extend(batch_labels.numpy())
        
    # Combine all batch arrays into unified large matrices
    return np.vstack(X_list), np.array(Y_list)

# ==========================================
# 4. RUN PIPELINE
# ==========================================
if __name__ == "__main__":
    # Adjust batch_size depending on your GPU memory (VRAM). 
    # For an RTX 4070 Ti (12GB) running Llama 3.2-3B in float16, a batch_size of 32 or 64 is ideal.
    BATCH_SIZE = 32 

    X_train, y_train = extract_features_batched("./data/logicnli_probing_targets-train.csv", batch_size=BATCH_SIZE, split_name="Train")
    X_dev, y_dev = extract_features_batched("./data/logicnli_probing_targets-dev.csv", batch_size=BATCH_SIZE, split_name="Dev")
    X_test, y_test = extract_features_batched("./data/logicnli_probing_targets-test.csv", batch_size=BATCH_SIZE, split_name="Test")
    
    print("\nTraining Linear Probe...")
    probe = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)
    probe.fit(X_train, y_train)
    print("Training complete.")
    
    print("\nEvaluating on Validation (Dev) set...")
    dev_preds = probe.predict(X_dev)
    print(f"Validation Macro F1-Score: {f1_score(y_dev, dev_preds, average='macro'):.4f}")
    
    print("\n=== FINAL TEST EVALUATION ===")
    test_preds = probe.predict(X_test)
    print(classification_report(y_test, test_preds, target_names=list(LABEL_MAP.keys())))
    
    hook_handle.remove()