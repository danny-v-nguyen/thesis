# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

*Authors:* Patrick Lewis, Ethan Perez, et. al.

Facebook AI Research; University College London, New York University

## Questions
- What is seq2seq?
  - Sequence-to-Sequence; transforms inputs to a sequence of encodings 
    and maps to a sequence of decoded outputs.

## Abstract

1. Introduce RAG models:
    - Parametric memory is pre-trained seq2seq model
    - Non-parametric memory is dense vector index of Wikipedia,
      accessed with pre-trained neural retriever.
2. Compared two RAG formulations
    - Conditions on retrieved passages across whole generated sequence
    - Can use different passages per token
3. Fine tune and evaluate models on wide range of knowledge-intensive
   NLP tasks; state of the art on three open domain QA tasks,  
   outperforming parametric seq2seq models and task-specific 
   retrieve-and-extract architectures.
4. For language generation, RAG models generate more specific, diverse
   and factual language than SOTA parametric-only seq2seq baseline
 
## Experiment

1. Open-domain Question Answering (QA)
    1. Natural Questions
    1. TriviaQA
    1. WebQuestions
    1. CuratedTrec
2. Abstractive Question Answering
    1. MSMARCO
3. Jeopardy Question Generation
    1. SearchQA
    1. SQuAD-tuned Q-BLEU-1 metric
4. Fact Verification
    1. FEVER

## Results

Good.