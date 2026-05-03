# A Survey on Automated Fact-Checking
*Authors:* Zhijiang Guo, Michael Schlichtkrull, Andreas Vlachos

University of Cambridge, UK

## Fact-Checking Framework

- Claim Detection
    - Identifying statements that require verification. This often 
    relies on "check-worthiness"—whether the public would be interested 
    in the truth of a claim.
- Evidence Retrieval
    - Searching for information (text, tables, or knowledge bases) that 
    supports or refutes the claim.
- Claim Verification
    - Assessing veracity based on retrieved evidence. This stage is 
    further divided into:
    - Verdict Prediction
        - Assigning a truthfulness label (e.g., True, False, Supported, 
        Refuted).
    - Justification Production
        - Generating explanations for the assigned verdict to increase 
        persuasiveness and transparency.

## Dataset Taxonomies

- **Natural vs. Artificial:** Datasets like Liar and MultiFC use real-world claims crawled from fact-checking sites. Conversely, FEVER uses artificial claims created by mutating Wikipedia sentences to provide better control over task complexity.
- **Input Types:** Inputs range from social media posts (e.g., PHEME, RumourEval) to political debate transcripts (e.g., ClaimRank) and structured subject-predicate-object triples.
- **Evidence Sources:** Most research assumes a single authoritative source like Wikipedia. Other datasets use semi-structured data like tables (TabFact) or specialized domains like science (SciFact) and public health (PUBHEALTH).

## Key-Modeling Strategies

- **Claim Detection:** Models often use Graph Neural Networks (GNNs) or 
LSTMs to capture social media propagation patterns and thread 
structures.  
- **Verification:** This is often modeled as Recognizing Textual 
Entailment (RTE). Modern systems use specialized components to aggregate 
and reason over multiple pieces of evidence.  
- **Justification**: Strategies include using attention weights to 
highlight salient evidence, logic-based systems for human-readable 
derivations, and summarization models to generate textual explanations.

## Critical Research Challenges

- **Faithfulness:** Abstractive models may generate "plausible" 
explanations that do not accurately reflect the model's actual reasoning 
(hallucinations).  
- **Dataset Biases:** Models often learn "artefacts" (e.g., specific 
indicative words) rather than the underlying task of verification.  
- **Multimodality & Multilinguality:** There is a need for larger 
annotated datasets that include images/video and cover languages beyond 
English.  
- **Subjectivity:** Check-worthiness is inherently subjective and varies 
by audience, recency, and geography.  
- **Proactive Fact-Checking:** Moving from "debunking" (reactive) to 
"pre-bunking" (proactive intervention).