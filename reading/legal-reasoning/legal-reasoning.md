# On Verifiable Legal Reasoning: A Multi-Agent Framework with Formalized Knowledge Representations
*Authors:* Albert Sadowski, Jaroslaw A. Chudziak

Warsaw University of Technology; 2025

## Summary: On Verifiable Legal Reasoning (SOLAR Framework)

### 1. Research Motivation
* **Performance Paradox**: Reasoning models (e.g., O1) are accurate but computationally expensive, while foundational models are efficient but fail at logical rigor in statutory analysis[cite: 2].
* **Objective**: To test if structured ontological representations can bridge the performance gap and improve consistency/explainability for foundational models[cite: 2].

### 2. Framework Architecture (SOLAR)
The framework decomposes legal reasoning into two distinct stages[cite: 2]:

#### Stage I: Knowledge Acquisition
* **Decomposition**: Specialized agents perform parallel analysis of statutory text[cite: 2].
* **Artifacts**: 
    * **TBox (Terminological Box)**: Captures legal classes (C), properties (P), and rules (R) in first-order logic[cite: 2].
    * **Interpreter**: A programmatic implementation (Python) of the calculation logic[cite: 2].
* **Validation**: An iterative loop ensures internal consistency and passes training sample evaluations[cite: 2].

#### Stage II: Knowledge Application
* **ABox Construction**: Maps specific case facts onto the TBox schema[cite: 2].
* **Symbolic Inference**: Uses an SMT solver to derive logically entailed conclusions from the ABox and TBox rules[cite: 2].
* **Answer Generation**: Combines inferred facts with the TBox interpreter to produce final numerical results[cite: 2].

### 3. Key Findings & Performance (SARA Numeric Dataset)
* **Accuracy Improvement**: Foundational models improved from an 18.8% (Zero-Shot) baseline to **76.4% accuracy** using SOLAR[cite: 2].
* **Gap Reduction**: The performance gap between reasoning and non-reasoning models narrowed from 68.2 to 5.9 percentage points[cite: 2].
* **Efficiency**: SOLAR achieved a ~45% reduction in token usage compared to Zero-Shot by passing compact TBox representations instead of full statutes[cite: 2].
* **Reliability**: Reduced performance variance ($\sigma=0.08$ vs $\sigma=0.28$ for zero-shot) suggests more predictable behavior[cite: 2].

### 4. Limitations & Challenges
* **Ontological Gaps**: Failures occurred when the TBox lacked specific vocabulary for critical concepts (e.g., itemized deductions)[cite: 2].
* **Latency**: Multi-agent overhead increased processing time (12.8s vs 1.5s for zero-shot)[cite: 2].
* **Grammar Patterns**: Difficulty in communicating co-occurrence requirements (e.g., needing both "Married" and "Joint" assertions for specific rules)[cite: 2].