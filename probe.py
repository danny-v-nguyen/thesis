import os
import pickle
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & LAYERS TO PROBE
# ==========================================
# We will probe these layers to see the logical evolution in the model's middle layers
TARGET_LAYERS = [10, 11, 12, 13, 14]

CACHE_DIR = "./probe_cache_multi"
os.makedirs(CACHE_DIR, exist_ok=True)

LABEL_MAP = {"fact": 0, "implication": 1, "negation": 2}

# Determine paths dynamically for each targeted layer
def get_feature_paths(split_name, layer):
    return {
        "X": os.path.join(CACHE_DIR, f"X_{split_name}_layer{layer}.npy"),
        "y": os.path.join(CACHE_DIR, f"y_{split_name}_layer{layer}.npy")
    }

# ==========================================
# 2. PYTORCH DATASET
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
# 3. MULTI-LAYER EXTRACTION ENGINE
# ==========================================
def extract_features_multi_layer(csv_path, batch_size=32, split_name="Split"):
    """
    Extracts features for all TARGET_LAYERS in parallel. 
    Resumes from cache if all target layer files already exist.
    """
    # Check if all layers are already cached for this split
    all_cached = True
    for layer in TARGET_LAYERS:
        paths = get_feature_paths(split_name, layer)
        if not (os.path.exists(paths["X"]) and os.path.exists(paths["y"])):
            all_cached = False
            break

    if all_cached:
        print(f"--> [All Layers Cached] Resuming [{split_name}] instantly from disk.")
        data_by_layer = {}
        for layer in TARGET_LAYERS:
            paths = get_feature_paths(split_name, layer)
            data_by_layer[layer] = (np.load(paths["X"]), np.load(paths["y"]))
        return data_by_layer

    print(f"--> [Extraction Needed] Initializing Llama 3.2-3B for [{split_name}] extraction...")
    
    global model, tokenizer, extracted_activations
    if 'model' not in globals():
        model_id = "meta-llama/Llama-3.2-3B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
        
        # Explicitly load model on GPU (fp16)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
        
        extracted_activations = {}
        
        # Dynamically create hooks for each target layer
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Ensure batch dims are preserved dynamically
                if isinstance(output, (tuple, list)):
                    extracted_activations[f"layer_{layer_idx}"] = output[0]
                else:
                    extracted_activations[f"layer_{layer_idx}"] = output
            return hook

        # Register forward hooks on all target layers
        for layer in TARGET_LAYERS:
            target_layer_module = model.model.layers[layer]
            target_layer_module.register_forward_hook(make_hook(layer))

    dataset = LogicNLIDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Storage arrays for each target layer
    X_lists = {layer: [] for layer in TARGET_LAYERS}
    Y_list = []
    
    progress_description = f"Extracting {split_name:<5}"
    for batch_sentences, batch_labels in tqdm(dataloader, desc=progress_description, unit="batch"):
        inputs = tokenizer(list(batch_sentences), padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model(**inputs)
            
        attention_mask = inputs["attention_mask"]
        
        # Perform masked mean pooling for every captured layer
        for layer in TARGET_LAYERS:
            layer_hidden_states = extracted_activations[f"layer_{layer}"]
            
            # Masked Mean-pooling calculations
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_states.size()).float()
            sum_embeddings = torch.sum(layer_hidden_states * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_pooled_batch = (sum_embeddings / sum_mask).detach().cpu().numpy()
            
            X_lists[layer].append(mean_pooled_batch)
            
        Y_list.extend(batch_labels.numpy())
        
    # Stack, save, and cache arrays for each layer
    results = {}
    y_final = np.array(Y_list)
    for layer in TARGET_LAYERS:
        X_final = np.vstack(X_lists[layer])
        paths = get_feature_paths(split_name, layer)
        
        np.save(paths["X"], X_final)
        np.save(paths["y"], y_final)
        results[layer] = (X_final, y_final)
        
    return results

# ==========================================
# 4. RUN PIPELINE & COMPARE
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 32 

    # Extract/Resume for all layers
    train_data = extract_features_multi_layer("./data/logicnli_probing_targets-train.csv", batch_size=BATCH_SIZE, split_name="Train")
    dev_data = extract_features_multi_layer("./data/logicnli_probing_targets-dev.csv", batch_size=BATCH_SIZE, split_name="Dev")
    test_data = extract_features_multi_layer("./data/logicnli_probing_targets-test.csv", batch_size=BATCH_SIZE, split_name="Test")
    
    # Safely free GPU VRAM before running ML classifiers
    if 'model' in globals():
        del model
        torch.cuda.empty_cache()

    # Results tracker
    layer_metrics = []

    print("\n--- Training and Evaluating Probes Across Layers ---")
    for layer in TARGET_LAYERS:
        print(f"Processing Layer {layer}...")
        
        X_tr, y_tr = train_data[layer]
        X_de, y_de = dev_data[layer]
        X_te, y_te = test_data[layer]
        
        # Train linear probe for this layer
        probe = LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=1000)
        probe.fit(X_tr, y_tr)
        
        # Save probe model to disk
        probe_path = os.path.join(CACHE_DIR, f"probe_layer_{layer}.pkl")
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)
            
        # Run evaluations
        dev_preds = probe.predict(X_de)
        test_preds = probe.predict(X_te)
        
        # Calculate per-class metrics
        class_f1s = f1_score(y_te, test_preds, average=None)  # [fact_f1, implication_f1, negation_f1]
        macro_f1 = f1_score(y_te, test_preds, average='macro')
        accuracy = accuracy_score(y_te, test_preds)
        
        layer_metrics.append({
            "Layer": layer,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Fact F1": class_f1s[0],
            "Implication F1": class_f1s[1],
            "Negation F1": class_f1s[2]
        })

    # Render results in a clean comparison table
    df_results = pd.DataFrame(layer_metrics)
    print("\n" + "="*65)
    print("                LAYER COMPARISON SUMMARY")
    print("="*65)
    print(df_results.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Macro F1": "{:.4f}".format,
        "Fact F1": "{:.4f}".format,
        "Implication F1": "{:.4f}".format,
        "Negation F1": "{:.4f}".format,
    }))
    print("="*65)