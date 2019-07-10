import matplotlib.pyplot as plt
import numpy as np

DQN = "../results/plotted_values/grid_dqnMiniGrid-Empty-6x6-v00maxrewards.npy"
POPULATION = 10
EVODQN = "../results/plotted_values/grid_evoreward_pop10MiniGrid-Empty-6x6-v01maxrewards.npy"

AVERAGED_OVER = 20
MAX_TRIALS_DQN = 336
MAX_TRIALS_EVODQN = 366

dqn_results = np.load(DQN)
evodqn_results = np.load(EVODQN)

#DQN
max_num_episodes_trial = MAX_TRIALS_DQN - AVERAGED_OVER
episode_reward_trials = np.load(DQN)
#episode_reward_trials = episode_reward_trials[:,:max_num_episodes_trial]
mean_rewards = np.average(episode_reward_trials, axis=0)
std_rewards = np.std(episode_reward_trials, axis=0)
print(np.max(std_rewards), mean_rewards[np.argmax(std_rewards)])
print(np.min(std_rewards))
plt.errorbar(
    range(1, mean_rewards.size + 1),
    mean_rewards,
    color='#4682B4FF',
    ecolor='#4682B455',
    errorevery=2,
    yerr=std_rewards,
    label="DQN")

#EVODQN
max_num_episodes_trial = MAX_TRIALS_EVODQN - AVERAGED_OVER
episode_reward_trials = np.load(EVODQN)
#episode_reward_trials = episode_reward_trials[:,:max_num_episodes_trial]
print(np.max(episode_reward_trials))
print(np.min(episode_reward_trials))
mean_rewards = np.average(episode_reward_trials, axis=0)
std_rewards = np.std(episode_reward_trials, axis=0)
plt.errorbar(
    range(1, mean_rewards.size + 1),
    mean_rewards,
    color='darkorange',
    ecolor='#FF8C0055',
    errorevery=2,
    yerr=std_rewards,
    label="DQN-evoReward, pop: " + str(POPULATION))

plt.xlabel("Episode")
plt.ylabel("Average Cumulative Reward")
plt.legend(title="DQN type")
plt.savefig("result.png")
