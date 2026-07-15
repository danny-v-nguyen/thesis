import os
import pickle
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & LAYERS TO PROBE
# ==========================================
TARGET_LAYERS = [10, 11, 12, 13, 14]

CACHE_DIR = "./probe_cache_multi"
os.makedirs(CACHE_DIR, exist_ok=True)

LABEL_MAP = {"fact": 0, "implication": 1, "negation": 2}

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
        
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="auto")
        
        extracted_activations = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, (tuple, list)):
                    extracted_activations[f"layer_{layer_idx}"] = output[0]
                else:
                    extracted_activations[f"layer_{layer_idx}"] = output
            return hook

        for layer in TARGET_LAYERS:
            target_layer_module = model.model.layers[layer]
            target_layer_module.register_forward_hook(make_hook(layer))

    dataset = LogicNLIDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    X_lists = {layer: [] for layer in TARGET_LAYERS}
    Y_list = []
    
    progress_description = f"Extracting {split_name:<5}"
    for batch_sentences, batch_labels in tqdm(dataloader, desc=progress_description, unit="batch"):
        inputs = tokenizer(list(batch_sentences), padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model(**inputs)
            
        attention_mask = inputs["attention_mask"]
        
        for layer in TARGET_LAYERS:
            layer_hidden_states = extracted_activations[f"layer_{layer}"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden_states.size()).float()
            sum_embeddings = torch.sum(layer_hidden_states * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_pooled_batch = (sum_embeddings / sum_mask).detach().cpu().numpy()
            X_lists[layer].append(mean_pooled_batch)
            
        Y_list.extend(batch_labels.numpy())
        
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
# 4. RUN PIPELINE & EVALUATE
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 32 

    # Extract/Resume Dataset Splits
    train_data = extract_features_multi_layer("./data/logicnli_probing_targets-train.csv", batch_size=BATCH_SIZE, split_name="Train")
    dev_data = extract_features_multi_layer("./data/logicnli_probing_targets-dev.csv", batch_size=BATCH_SIZE, split_name="Dev")
    test_data = extract_features_multi_layer("./data/logicnli_probing_targets-test.csv", batch_size=BATCH_SIZE, split_name="Test")
    
    if 'model' in globals():
        del model
        torch.cuda.empty_cache()

    # We use Layer 12's shapes as standard inputs for the baseline classifiers
    # (Since baselines ignore feature values, their shapes simply need to match y_test)
    sample_layer = TARGET_LAYERS[0]
    X_tr_sample, y_train = train_data[sample_layer]
    X_te_sample, y_test = test_data[sample_layer]

    # Tracker list for final comparison table
    report_rows = []

    # ------------------------------------------
    # A. COMPUTE BASELINES (Pure Random Chance)
    # ------------------------------------------
    print("\nComputing statistical baselines...")
    baselines = {
        "Baseline: Uniform Random": DummyClassifier(strategy="uniform", random_state=42),
        "Baseline: Majority Class": DummyClassifier(strategy="most_frequent", random_state=42),
        "Baseline: Stratified Random": DummyClassifier(strategy="stratified", random_state=42)
    }

    for name, dummy_model in baselines.items():
        # Fit baseline on training labels, then predict on test labels
        dummy_model.fit(X_tr_sample, y_train)
        preds = dummy_model.predict(X_te_sample)
        
        class_f1s = f1_score(y_test, preds, average=None, zero_division=0)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        accuracy = accuracy_score(y_test, preds)
        
        report_rows.append({
            "Source": name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Fact F1": class_f1s[0],
            "Implication F1": class_f1s[1],
            "Negation F1": class_f1s[2]
        })

    # ------------------------------------------
    # B. EVALUATE LAYER-BY-LAYER PROBES
    # ------------------------------------------
    print("\nTraining and evaluating probing classifiers...")
    for layer in TARGET_LAYERS:
        X_tr, y_tr = train_data[layer]
        X_te, y_te = test_data[layer]
        
        # Train probe for current layer
        probe = LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=5000)
        probe.fit(X_tr, y_tr)
        
        # Save probe model to disk
        probe_path = os.path.join(CACHE_DIR, f"probe_layer_{layer}.pkl")
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)
            
        preds = probe.predict(X_te)
        
        class_f1s = f1_score(y_te, preds, average=None)
        macro_f1 = f1_score(y_te, preds, average='macro')
        accuracy = accuracy_score(y_te, preds)
        
        report_rows.append({
            "Source": f"Llama Layer {layer}",
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Fact F1": class_f1s[0],
            "Implication F1": class_f1s[1],
            "Negation F1": class_f1s[2]
        })

    # ------------------------------------------
    # C. PRINT UNIFIED REPORT CARD
    # ------------------------------------------
    df_results = pd.DataFrame(report_rows)
    print("\n" + "="*80)
    print("                    PROBING RESULTS VS. BASELINES REPORT")
    print("="*80)
    print(df_results.to_string(index=False, formatters={
        "Accuracy": "{:.4%}".format,
        "Macro F1": "{:.4%}".format,
        "Fact F1": "{:.4%}".format,
        "Implication F1": "{:.4%}".format,
        "Negation F1": "{:.4%}".format,
    }))
    print("="*80)