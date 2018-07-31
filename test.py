import numpy as np
import matplotlib.pyplot as plt
import gym
import math

AVERAGED_OVER = 20

max_num_episodes_trial = 366-AVERAGED_OVER
episode_reward_trials = np.load("plotted_values/grid_evoreward_pop10MiniGrid-Empty-6x6-v01.npy")
print (episode_reward_trials)
