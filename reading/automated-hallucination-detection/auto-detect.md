# (Im)possibility of Automated Hallucination Detection in Large Language Models
*Authors:* Anonymous Authors

*Note:* Preliminary work. Under review by International Conference on ICML 2025
        Workshop on Reliable and Responsible Foundation Models.

## Summary

Theoretical framework characterizing automatic detection of hallucinations 
produced by LLMs. Mathematical definitions of what the `hallucination detection`
game entails, and frames it as a language identification process. Lots of references
to Gold-Angluin Framework for language identification w/ Kleinberg & Mullainathan 
adaptation for language generation.

## Abstract

- Investigates whether an algorithm can be trained to determine whether or not
  a model's output contains hallucinations.
- Establish equivalence between hallucination detection and language identification
    - Any algorithm that can identify languages (in the limit) can be transformed 
      into one that detects hallucinations
    - Hallucination detection is largely impossible for most L*
- Second; enriching training data w/ both positive and negative examples make
  hallucination detection possible for any L*
- Importance of expert labeling in practical deployment of LLMs; Reinforcement Learning
  with Human Feedback (RLHF) is popular for a reason

## Introduction

- LLMs frequently hallucinate (Ji et al., 2023)
- Kalai & Vempala, 2024; Kalavasis et al., 2024; theoretical benefit from negative
  examples in training
- RLF leverage negative examples to reduce hallucinations/increase model reliability
  (Yang et al.,2024) (DeepMind, 2024)

### Related Work

- Gold-Angluin Framework
    - Kleinberg & Mullainathan adaptation
- Li et al., 2025; extend perspective to when "uniform" and "non-uniform" language
  generation is achievable
- Kalai & Vempala, 2024 connects *calibration* to increased hallucination rates
- Peng et al., 2024; identified fundamental limits to transformer architectures
    + (Chen et al., 2024)
    + (Guan et al., 2024) 
    + Function composition underlies reasoning; transformers cannot compose functions
      when the domain is 'too large'
- Xu et al., 2024; leverage complexity theory to demonstrate that hallucinations
  are inevitable (under assumptions)

### Empirical Work on Hallucination Detection

- SelfCheckGPT (Manakul et al., 2024): stochastic sampling of model responses
- Azaria & Mitchell, 2023; internal states of LLM to classify fact v. not
    - Significantly better than probability methods
    - *Me:* Does this mean that the models are already correctly quantifying
            the uncertainty associated with generated responses?
- Comprehensive surveys on hallucination detection; read:
    - Ji et al., 2023
    - Zhang et al., 2023

## Model

*Me:* Honestly, it's a lot to get into and I currently find it dubious as to 
whether or not it's worth re-transcribing exactly. Revisit if necessary.

- Define languages L, domains X, adversary and learner.
- Adversary is the LLM with responses that targets some language K in L, and
  a generated subset of statements $E_k$ mapping to a domain G that may or not be
  a subset of the domain X.
- G is contains hallucinations if it contains elements outside of K.
- The learner witnesses each statement in E one at a time
    - Check if the statement is in G
    - Guess if it's true or not, $g_t$
- Learner detects hallucinations "in the limit" if:
    - for all K in L, E of K, and candidate G in X
    - it holds that there exists some $t_0$ in $N : g_t = 1{G \subset N }$ for all $t > t_0$
- TL;DR: L captures different "worlds", X elements are all possible "statements"

## Overview of the Approach

- There is a way to define an algorithm that can detect hallucinations...
