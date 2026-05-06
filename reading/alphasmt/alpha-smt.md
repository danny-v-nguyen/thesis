# AlphaSMT: A Reinforcement Learning Guided SMT Solver
*Author:* Zhengyang Lu

2023; University of Waterloo

## Overview
AlphaSMT is an adaptive, reinforcement learning (RL) based SMT solver built on Z3. 
It addresses the **tactic selection problem** by dynamically constructing the 
optimal sequence of reasoning steps (tactics) for a given input formula[cite: 3].

## Key Methodology
* **MDP Framework**: Formalizes tactic selection as a Markov Decision Process 
  where the agent chooses tactics based on the current formula state and solving 
  history[cite: 3].
* **Deep MCTS**: Integrates Monte-Carlo Tree Search with Deep Neural Networks. 
  MCTS acts as a lookahead planning step during training to improve policy 
  decisions[cite: 3].
* **Neural Network Architecture**: Uses a dual-head DNN (policy and value) 
  incorporating Transformer layers to capture temporal relationships in tactic 
  history and action embeddings to understand tactic "language"[cite: 3].
* **Training vs. Runtime**: The expensive MCTS lookahead is used only during 
  offline training on representative benchmarks; at runtime, the solver uses 
  the raw DNN policy for rapid decision-making[cite: 3].

## Experimental Results
The solver was evaluated on SMT-LIB benchmarks across three logics (QF_NIA, 
QF_NRA, QF_BV) with a 300s timeout[cite: 3]:

* **Performance**: AlphaSMT significantly outperformed the default Z3 solver, 
  solving up to **80.5% more instances** in QF_NRA testing sets[cite: 3].
* **Tactic Timeout**: Longer tactic timeouts generally improved the success 
  rate[cite: 3].
* **Pre-solver**: Implementing a 10s Z3 pre-solving stage reduced overhead for 
  easy instances and expedited overall solving time[cite: 3].
* **Adaptability**: The agent effectively switches strategies if a specific 
  tactic sequence fails or times out[cite: 3].

## Contributions & Future Work
* **Adaptiveness**: Unlike previous tools (e.g., FastSMT), AlphaSMT constructs 
  strategies dynamically rather than picking from fixed candidate sets[cite: 3].
* **Future Directions**: Plans include cross-solver tactic integration, online 
  learning (updating the policy during runtime), and leveraging internal solver 
  data from failed/timed-out tactic attempts[cite: 3].