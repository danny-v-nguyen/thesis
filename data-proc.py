import json
import csv
import os

def generate_probing_dataset(logic_path, language_path, output_csv_path):
    # 1. Load both JSON files
    print("Loading datasets...")
    with open(logic_path, "r", encoding="utf-8") as f:
        logic_data = json.load(f)
        
    with open(language_path, "r", encoding="utf-8") as f:
        language_data = json.load(f)
        
    csv_rows = []
    
    # 2. Iterate through matched entries
    for entry_id, logic_node in logic_data.items():
        # Safeguard: Ensure the entry exists in the language file
        if entry_id not in language_data:
            continue
        
        lang_node = language_data[entry_id]
        
        # --- EXTRACT FACTS ---
        # LogicNLI language facts map 1-to-1 by index to the logic manifestations
        lang_facts = lang_node.get("facts", [])
        for fact_text in lang_facts:
            csv_rows.append({
                "Sentence": fact_text.strip(),
                "Label": "fact"
            })
            
        # --- EXTRACT IMPLICATIONS ---
        # Check rule metadata in logic to filter corresponding text in language
        logic_rules = logic_node.get("rules", {})
        lang_rules = lang_node.get("rules", [])
        
        for r_idx_str, r_meta in logic_rules.items():
            r_idx = int(r_idx_str)
            # Ensure index safety between both files
            if r_idx < len(lang_rules):
                if r_meta.get("type") == "imp":
                    csv_rows.append({
                        "Sentence": lang_rules[r_idx].strip(),
                        "Label": "implication"
                    })
                    
        # --- EXTRACT NEGATIONS ---
        # Look inside the statements structure for any element containing "-"
        logic_stmts = logic_node.get("statements", {})
        lang_stmts = lang_node.get("statements", [])
        
        for s_idx_str, s_array in logic_stmts.items():
            s_idx = int(s_idx_str)
            if s_idx < len(lang_stmts):
                # Check if the negative polarity flag "-" exists anywhere in the array
                if "-" in s_array:
                    csv_rows.append({
                        "Sentence": lang_stmts[s_idx].strip(),
                        "Label": "negation"
                    })

    # 3. Serialize output to a clean CSV file
    print(f"Writing {len(csv_rows)} rows to {output_csv_path}...")
    with open(output_csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Sentence", "Label"])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print("Dataset generation complete.")

# --- EXECUTION ---
if __name__ == "__main__":
    # Update these paths to match your local file names/directories
    DATA_ROOT = "./data"
    LOGICNLI_ROOT = f"{DATA_ROOT}/LogicNLI/dataset/LogicNLI_sim"
    TRAIN_LOGIC_FILE = f"{LOGICNLI_ROOT}/train_logic.json"
    TRAIN_LANGUAGE_FILE = f"{LOGICNLI_ROOT}/train_language.json"
    OUTPUT_CSV = f"{DATA_ROOT}/logicnli_probing_targets-train.csv"
    
    # Simple check to help you debug file placement
    if os.path.exists(TRAIN_LOGIC_FILE) and os.path.exists(TRAIN_LANGUAGE_FILE):
        generate_probing_dataset(TRAIN_LOGIC_FILE, TRAIN_LANGUAGE_FILE, OUTPUT_CSV)
    else:
        print("Error: Please make sure 'train_logic.json' and 'train_language.json' are in this directory.")

    TEST_LOGIC_FILE = f"{LOGICNLI_ROOT}/test_logic.json"
    TEST_LANGUAGE_FILE = f"{LOGICNLI_ROOT}/test_language.json"
    OUTPUT_CSV = f"{DATA_ROOT}/logicnli_probing_targets-test.csv"
    
    # Simple check to help you debug file placement
    if os.path.exists(TEST_LOGIC_FILE) and os.path.exists(TEST_LANGUAGE_FILE):
        generate_probing_dataset(TEST_LOGIC_FILE, TEST_LANGUAGE_FILE, OUTPUT_CSV)
    else:
        print("Error: Please make sure 'test_logic.json' and 'test_language.json' are in this directory.")
    
    DEV_LOGIC_FILE = f"{LOGICNLI_ROOT}/dev_logic.json"
    DEV_LANGUAGE_FILE = f"{LOGICNLI_ROOT}/dev_language.json"
    OUTPUT_CSV = f"{DATA_ROOT}/logicnli_probing_targets-dev.csv"
    
    # Simple check to help you debug file placement
    if os.path.exists(DEV_LOGIC_FILE) and os.path.exists(DEV_LANGUAGE_FILE):
        generate_probing_dataset(DEV_LOGIC_FILE, DEV_LANGUAGE_FILE, OUTPUT_CSV)
    else:
        print("Error: Please make sure 'dev_logic.json' and 'dev_language.json' are in this directory.")