import matplotlib.pyplot as plt
import numpy as np


DQN = "plotted_values/dqn_rewards.npy"
POPULATION = 20
EVODQN = "plotted_values/evoreward_pop" + str(POPULATION) + "_rewards.npy"

AVERAGED_OVER = 20
MAX_TRIALS_DQN = 336
MAX_TRIALS_EVODQN = 366

dqn_results = np.load(DQN)
evodqn_results = np.load(EVODQN)

#DQN
max_num_episodes_trial = MAX_TRIALS_DQN-AVERAGED_OVER
episode_reward_trials = np.load(DQN)
episode_reward_trials = episode_reward_trials[:,:max_num_episodes_trial]
mean_rewards = np.average(episode_reward_trials,axis=0)    
std_rewards = np.std(episode_reward_trials,axis=0)
plt.errorbar(range(1,mean_rewards.size+1),mean_rewards,color='#4682B4FF',ecolor='#4682B455',errorevery=5,yerr=std_rewards,label="DQN")

#EVODQN
max_num_episodes_trial = MAX_TRIALS_EVODQN-AVERAGED_OVER
episode_reward_trials = np.load(EVODQN)
episode_reward_trials = episode_reward_trials[:,:max_num_episodes_trial]
mean_rewards = np.average(episode_reward_trials,axis=0)    
std_rewards = np.std(episode_reward_trials,axis=0)
plt.errorbar(range(1,mean_rewards.size+1),mean_rewards,color='darkorange',ecolor='#FF8C0055',errorevery=5,yerr=std_rewards,label="DQN-evoReward, pop: "+str(POPULATION))
             
plt.xlabel("Episode")
plt.ylabel("Average Cumulative Reward")
plt.legend(title="DQN type")
plt.savefig("result.png")