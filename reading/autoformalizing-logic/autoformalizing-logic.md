# Autoformalizing Natural Language to First-Order Logic: A Case Study in Logical Fallacy Detection

*Authors:* Abhinav Lalwani, Tasha Kim, Lovish Chopra, et. al.

Stanford University, X, moonshot factory, Max Planck Institute for Intelligent 
Systems, ETH Zurich, University of Toronto

## Questions
- What is SMT?
  - Satisfiability Modulo Theory: From a set of premises and a claim, is there 
    any combination that satisfies the claim? Or disproves it I guess.

## Abstract

- NL translation to First-Order Logic (FOL) is a foundational challenge in NLP
- Applications in reasoning, misinformation tracking, knowledge validation
- NL2FOL: framework to autoformalize natural language to FOL using LLMs
- Satisfiability Modulo Theory (SMT) solvers to reason about logical validity

## Overview

- Module A: NL2FOL
    1. Input Sentence -> **Claim and implication Parser** -> Output Claim, Implications
    2. Claims, Implications -> **Entity Extractor** -> Referring Expressions
    3. Referring Expressions -> **Entity Relation Classifier** -> Entity Relations
    4. Entity Relations -> **Property Extractor** -> Properties
    5. Properties -> **Background Knowledge Retriever** -> Background Knowledge
    - Everything Above -> **FOL Formulation Engine** -> FOL Formula
- Module B: FOL to SMT
    1. FOL Formula, Background Knowledge -> **SMT Compiler** -> SMT File
- Module C: Interpreting SMT Results
    1. SMT File -> **CVC SMT Solver** -> Counter-Model
    2. Counterexample -> **Semantic Counter-Model Decipherer** -> Counterexample

TL;DR, take sentence and generate a FOL formula through various breakdowns to feed 
through an SMT solver. Afterwards, use an LLM to explain the SMT results.

## Conclusion

- Approach seems to work for short premises; misses complex relational constructs
- Hard to tell if this generalizes to other tasks and domains
- Does this extend to higher-order logic?
- Computational cost: high-performance GPU cost for LLM inference, CPU cost for 
  SMT solvers, and really high API usage for GPT/Llama