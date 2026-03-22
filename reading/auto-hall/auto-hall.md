# AutoHall: Automated Factuality Hallucination Dataset Generation for Large Language Models 
*Authors:* Zhouying Cao,
           Yifei Yang,
           Xiaojing Li,
           Hai Zhao

*Source URL:* [https://arxiv.org/html/2310.00259v3]()

27 November 2025

## Summary

`AutoHall` method to **Auto**matically construct model-specific **Hall**ucination 
datasets based on existing fact-checking datasets.

## Methodology

### LLM Factuality Hallucination Formulation
Obvious definition; iff a response does not agree with fact database, it's a 
hallucination

### Automatic Generation of Factuality Hallucination Datasets
Given:

- Existing fact database make ground truth claims
  - Climate-fever [51]
  - PUBHEALTH [52]
  - WICE [53]

1. Reference Generation
    - Prompt model to generate a reference for a given claim (may be known true or false)
    - Responses that contained "I cannot provide a specific reference for the claim
      you mentioned..." and likewise were discarded
2. Claim Classification
    - Use model to perform claim classification: does the generated reference support 
      or refute a given claim?
    - Apparently LLM binary classification is pretty good? [11,47,48]
3. Hallucination Collection
    - String match algorithm (?) to collect hallucination dataset
    - If classification result does not match ground truth label; it is a 
      hallucination
    - Sample factual references equally with hallucinated ones

### Hallucination Detection Approach

1. Take original claim plus the reference (from the generated dataset)
2. Prompt `k` times independently to provide a reference for the original claim
3. Check for contradiction within `k` against the original claim and reference.
    - Note that the LLM is used to detect the contradictions

## Experiment

- Models
    - ChatGPT Turbo 3.5
    - GPT-4o
    - Open-source: server w/ 2x nVidia A100 GPU 80GB Memory
        - Llama2-7B-Chat
        - Llama2-13B-Chat
        - Llama3-8B-Instruct
- Statistics
    - Everything hovers around [16-25]% regardless of temperature or model
    - WICE responses are higher, [26-30]% except for GPT-4o still closer to
      [20-22]%
- Comparisons
    - Slightly better than other self-consistency methods
- Neat stuff
    - Hallucinations seemed to correlate to topics depending on model, signifying
      training set differences (probably)