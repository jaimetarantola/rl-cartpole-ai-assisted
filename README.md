# RL CartPole (PyTorch) — AI-Assisted Take-Home Style Project

## AI-Assisted Development Disclosure

This project was developed with the assistance of AI tools, similar to modern industry workflows.  
All design decisions and final code were reviewed and validated by the author.


## Original Project Prompt

You are given approximately **2 hours** to complete a small machine learning task using **AI assistance tools if desired**.

**Task:**  
Implement a minimal **reinforcement learning agent** to solve the **CartPole-v1** environment using **Python and PyTorch**.

**Constraints**
- Favor clarity and readability over performance
- Avoid unnecessary abstractions
- No advanced RL frameworks (PPO/DQN libraries)

**Deliverables**
- Clean GitHub repository
- Simple policy network
- Training script that logs rewards
- Plot showing learning progress
- README explaining approach and possible improvements

---

## Overview

This repository implements a **vanilla policy-gradient (REINFORCE)** agent trained on the CartPole-v1 environment using PyTorch.

The goal is not state-of-the-art performance, but a **clear, reviewable implementation** suitable for an interview or take-home assignment.

---

## Project Structure

```text
rl-cartpole-ai-assisted/
├── src/
│   ├── env.py
│   ├── policy.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
├── results/
├── requirements.txt
└── README.md

