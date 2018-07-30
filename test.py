import numpy as np
import matplotlib.pyplot as plt
import gym
import math

AVERAGED_OVER = 20

max_num_episodes_trial = 366-AVERAGED_OVER
episode_reward_trials = np.load("plotted_values/evoreward_pop20_rewards.npy")
episode_reward_trials = episode_reward_trials[:,:max_num_episodes_trial]
mean_rewards = np.average(episode_reward_trials,axis=0)    
std_rewards = np.std(episode_reward_trials,axis=0)

plt.errorbar(range(1,mean_rewards.size+1),mean_rewards,yerr=std_rewards,label="DQN-evoReward, pop: "+str(20))

plt.xlabel("Episode")
plt.ylabel("Average Cumulative Reward")
plt.legend(title="DQN type")
plt.savefig("test.png")
