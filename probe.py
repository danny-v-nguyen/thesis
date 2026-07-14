import os
import pickle
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Directory where intermediate extracted features and final probe will be stored
CACHE_DIR = "./probe_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PROBE_SAVE_PATH = os.path.join(CACHE_DIR, "logistic_regression_probe.pkl")

# File paths for saving/resuming extracted features
FEATURE_PATHS = {
    "Train": {
        "X": os.path.join(CACHE_DIR, "X_train.npy"),
        "y": os.path.join(CACHE_DIR, "y_train.npy")
    },
    "Dev": {
        "X": os.path.join(CACHE_DIR, "X_dev.npy"),
        "y": os.path.join(CACHE_DIR, "y_dev.npy")
    },
    "Test": {
        "X": os.path.join(CACHE_DIR, "X_test.npy"),
        "y": os.path.join(CACHE_DIR, "y_test.npy")
    }
}

LABEL_MAP = {"fact": 0, "implication": 1, "negation": 2}

# ==========================================
# 2. PYTORCH BATCHING DATASET
# ==========================================
class LogicNLIDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).dropna(subset=["Sentence"])
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
    """
    Extracts features or resumes from a saved .npy checkpoint if it exists.
    """
    x_cache_path = FEATURE_PATHS[split_name]["X"]
    y_cache_path = FEATURE_PATHS[split_name]["y"]

    # --- RESUME / CHECKPOINT CHECK ---
    if os.path.exists(x_cache_path) and os.path.exists(y_cache_path):
        print(f"--> Found cached features for [{split_name}]. Loading from disk to resume instantly...")
        X = np.load(x_cache_path)
        y = np.load(y_cache_path)
        return X, y

    # If no cache exists, load model infrastructure dynamically to preserve VRAM/RAM until needed
    print(f"--> No cache found for [{split_name}]. Initializing model for extraction...")
    
    global model, tokenizer, extracted_activations
    if 'model' not in globals():
        print("Loading Llama 3.2-3B...")
        model_id = "meta-llama/Llama-3.2-3B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
        
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        
        extracted_activations = {}
        def residual_stream_hook(module, input, output):
            if isinstance(output, (tuple, list)):
                extracted_activations["hidden_states"] = output[0]
            else:
                extracted_activations["hidden_states"] = output

        target_layer = model.model.layers[12]
        target_layer.register_forward_hook(residual_stream_hook)

    dataset = LogicNLIDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    X_list = []
    Y_list = []
    
    progress_description = f"Extracting {split_name:<5}"
    
    for batch_sentences, batch_labels in tqdm(dataloader, desc=progress_description, unit="batch"):
        inputs = tokenizer(
            list(batch_sentences), 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model(**inputs)
            
        layer_hidden_states = extracted_activations["hidden_states"]
        attention_mask = inputs["attention_mask"]
        
        # Masked mean-pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_states.size()).float()
        sum_embeddings = torch.sum(layer_hidden_states * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled_batch = (sum_embeddings / sum_mask).detach().cpu().numpy()
        
        X_list.append(mean_pooled_batch)
        Y_list.extend(batch_labels.numpy())
        
    X = np.vstack(X_list)
    y = np.array(Y_list)

    # Save features to disk to enable resuming later
    print(f"Saving extracted [{split_name}] features to disk...")
    np.save(x_cache_path, X)
    np.save(y_cache_path, y)
    
    return X, y

# ==========================================
# 4. RUN PIPELINE
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 32 

    # 1. Extract/Resume Dataset Splits
    X_train, y_train = extract_features_batched("./data/logicnli_probing_targets-train.csv", batch_size=BATCH_SIZE, split_name="Train")
    X_dev, y_dev = extract_features_batched("./data/logicnli_probing_targets-dev.csv", batch_size=BATCH_SIZE, split_name="Dev")
    X_test, y_test = extract_features_batched("./data/logicnli_probing_targets-test.csv", batch_size=BATCH_SIZE, split_name="Test")
    
    # Clean up model from GPU memory to make room for downstream work
    if 'model' in globals():
        del model
        torch.cuda.empty_cache()

    # 2. Train and Serialize the Probe
    print("\nTraining Linear Probe...")
    probe = LogisticRegression(l1_ratio=0, max_iter=10000)
    probe.fit(X_train, y_train)
    print("Training complete.")

    # Save the trained probe model to disk
    with open(PROBE_SAVE_PATH, "wb") as f:
        pickle.dump(probe, f)
    print(f"Saved trained probe classifier to: {PROBE_SAVE_PATH}")
    
    # 3. Evaluate
    print("\nEvaluating on Validation (Dev) set...")
    dev_preds = probe.predict(X_dev)
    print(f"Validation Macro F1-Score: {f1_score(y_dev, dev_preds, average='macro'):.4f}")
    
    print("\n=== FINAL TEST EVALUATION ===")
    test_preds = probe.predict(X_test)
    print(classification_report(y_test, test_preds, target_names=list(LABEL_MAP.keys())))