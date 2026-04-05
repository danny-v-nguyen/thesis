# Olympiad-level formal mathematical reasoning with reinforcement learning
*Authors:* Way too many people; Hubert et al. Google DeepMind London

- AlphaProof core:
  1. Lean (formal math proof language) environment. Agent observes a theorem to 
  be proven, call this a 'state'
  2. Proof Network: take current Lean state and output $N$ promising tactics with
  an estimated difficulty statistic of difficulty to prove
  3. Tree search: use the proof network output to explore potential proof paths.
  This search is a specialized implementation.

## Training

- 3 billion parameter encoder-decoder transformer model used to interpret an input
  Lean state to produce the promising next tactics and associated provability values.
- 300 billion tokens of Lean code and math text used to train this model for
  next-token prediction like typical LLM methods
- Supervised fine tuning; 300,000 state-tactic pairs extracted from human proofs
  in the Mathlib library
- Auto-formalization process converted approximately 1 million language problems
  into ~80 million formal Lean proofs to use as training data.

## Methods

### Interactive theorem proving

- Formal and verifiable proofs using proof assistants via Isabelle, HOL Light,
  Rocq and Lean.

### Machine learning for theorem proving

- Formalized mathematics benchmarks MiniF2F, ProofNet, PutnamBench, LISA or HOList.
- Baldur pioneered whole-proof generation using LLM
- ref.46 approach:
  - Draft, Sketch, Prove
  - This line extended by work on subgoal decomposition
- Sophisticated search strategies for neural theorem proving
  - HyperTree Proof Search
  - Variants of Monte Carlo
- Kimina and DeepSeek-Prover explored interleaving natural-language and formal proofs

