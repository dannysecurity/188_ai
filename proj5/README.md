# Project 5

This repository contains my completed implementation of Project 5 from UC Berkeley’s CS 188: Introduction to Artificial Intelligence. The project focused on probabilistic inference in dynamic, partially observable environments—specifically, tracking and locating ghosts using Bayes nets, exact inference, and particle filtering.

My implementations span:
- Exact inference via variable elimination and the forward algorithm
- Approximate inference via particle filtering
- Greedy decision-making based on belief distributions

---

## \Core Concepts Implemented

All logic is implemented across the following files:

- `inference.py`:  
  - Constructed Bayes nets and factors
  - Implemented exact inference updates (`observeUpdate`, `elapseTime`)
  - Implemented particle filtering for approximate belief tracking
  - Handled special cases (e.g., zero-weight resampling, jail positions)
  - Extended `DiscreteDistribution` class (normalization, sampling)

- `factorOperations.py`:  
  - Implemented fundamental factor operations: `joinFactors`, `eliminate`, and `normalize`
  - Supported general-purpose factor manipulation used in variable elimination

- `bustersAgents.py`:  
  - Developed the GreedyBustersAgent to choose actions based on most likely ghost positions
  - Integrated inference modules with agent decision-making for game-playing

---

## 🔍 Inference Techniques

- **Bayes Nets**: Built network structures for ghost tracking, including defining variable domains and dependencies.
- **Exact Inference**: Used forward algorithm to maintain and update belief distributions with time and observations.
- **Variable Elimination**: Implemented an efficient form of inference that interleaves joins and marginalization.
- **Approximate Inference**: Implemented particle filtering, including initialization, reweighting with observations, and prediction through sampling.


