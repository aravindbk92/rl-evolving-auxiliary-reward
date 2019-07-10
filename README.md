# DQN with Evolving Auxilliary Reward Function

DQN is augmented with a reward function (**EvoReward**) that evolves (each episode currently) using a basic genetic algorithm. 

EvoReward takes in the current environment state and agent action to provide a good direction for the agent, even in the absence of external rewards. Experiments with Q-learning and Deep Q-learning agents augmented with EvoReward show that it significantly improves performance in both dense and sparsely rewarded environments and that it has the potential to be
practically used.

# Requirements
python 3.5
Keras 
gym
gym_minigrid
numpy
matplotlib

[gym_minigrid](https://github.com/maximecb/gym-minigrid) is a minimal gym environment with sparse rewards. 

# How to use
Run: python `paralell_trials.py`
Runs DQN+EvoReward by default with a population of 10 on MiniGrid-Empty-6x6-v0 environment.
Use '--dqn_type 0' argument to use a normal DQN to compare.
Use '--help' to see more options.

# Files
- `evoDQN.py`: DQN implementation supporting EvoReward function
- `evoReward.py`: Evolving reward function class
- `multiprocess.py`: For multiprocessing each individual in the population. **This is test code.** Does not work with EvoReward currently.
- `parallel_trials.py`: Aggregate results from multiple trials of training.
- `scripts/run_grid_trials`: Script to run parallel_trials.py in multiple processes to get results faster.

# Results
Formal algorithm and results can be found in [docs](docs/EvoReward__Deep_Q_Network_with_Evolving_Augmented_Rewards.pdf)

