# Reinforcement Learning - Rainbow DQN

This repository contains an implementation of **Rainbow DQN** applied to the **CartPole environment** using PyTorch and OpenAI Gym. 

## 📄 Paper Reference
This implementation is based on the **Rainbow DQN** paper:  
["Rainbow: Combining Improvements in Deep Reinforcement Learning"](https://arxiv.org/abs/1710.02298) by Hessel et al.

A **detailed PDF explanation of the paper** is provided in this repository.

## 📌 Methods Implemented
The **Rainbow DQN** algorithm combines several improvements over the standard DQN, including:
- **Double DQN** (Prevents overestimation of Q-values)
- **Dueling Networks** (Separates state value & action advantage functions)
- **Prioritized Experience Replay (PER)** (Replays more important transitions more frequently)
- **Multi-step Learning (n-step returns)** (Encourages faster credit assignment)
- **Distributional RL (Categorical DQN)** (Predicts full return distributions instead of Q-values)
- **Noisy Networks** (For exploration instead of epsilon-greedy)
- **Target Network Updates** (For stable learning)

For comparison, we also provide a **Vanilla DQN** implementation.

## 📊 Results on CartPole
The notebook outputs two result images comparing **Rainbow DQN** with **Vanilla DQN**:

### **Rainbow DQN Performance**
![Rainbow DQN](images/rainbow_cartpole.png)

### **Vanilla DQN Performance**
![Vanilla DQN](images/dqn_cartpole.png)

### **Observations**
- Rainbow DQN achieves a **perfect score (200)** quickly and stabilizes.
- Vanilla DQN struggles with **high variance and instability**, failing to converge consistently.
