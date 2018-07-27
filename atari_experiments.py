import matplotlib.pyplot as plt
from atari import AtariTrain
import numpy as np
import math

NUM_TRIALS = 2
NUM_EPISODES = 1000
SOLVED_REWARD_CRITERIA = 195.0

DQN_TYPE = 0
EVODQN_TYPE = 1

atari = AtariTrain(env="CartPole-v0", solve_score=SOLVED_REWARD_CRITERIA)

def run_dqn_trials(dqn_type=DQN_TYPE, num_population=40):
    episode_reward_trials = np.empty((0,NUM_EPISODES))
    episodes_to_solve = np.array([])
    if dqn_type == EVODQN_TYPE:
        print ("----DQN with augmented reward----")    
    else:
        print ("----DQN with normal reward----")
    
    max_num_episodes_trial = NUM_EPISODES
    for trial in range(NUM_TRIALS):
        print()
        print ("trial:", trial)
        
        if dqn_type == EVODQN_TYPE:
            [episode_rewards,augment] = atari.evodqn_train(n_episodes=NUM_EPISODES, n_population=num_population)
        else:
            [episode_rewards] = atari.dqn_train(n_episodes=NUM_EPISODES)
        
        # find in which episode it was solved for current trial
        episodes_to_solve = np.append(episode_rewards.size)

        # find maximum number of episodes reached across trials
        if (episode_rewards.size < max_num_episodes_trial):
            max_num_episodes_trial = episode_rewards.size
            
        # append reward history from each trial
        np.pad(episode_rewards,(0,NUM_EPISODES-episode_rewards.size),mode='constant',constant_values=SOLVED_REWARD_CRITERIA)
        episode_reward_trials = np.append(episode_reward_trials,[episode_rewards],axis=0)
    
    # find mean and std
    episode_reward_trials = episode_reward_trials[:,:max_num_episodes_trial]
    mean_rewards = np.average(episode_reward_trials,axis=0)    
    std_rewards = np.std(episode_reward_trials,axis=0)
    
    # print stats
    with open('log.txt', 'w') as f:
        print (file=f)
        print ("--DQN stats--") if (dqn_type==DQN_TYPE) else print ("--DQN with evoRewards stats--",file=f)
        print (">> Episodes taken to solve (reach score: 195.0):",file=f)
        print ("Mean:", np.average(episodes_to_solve),file=f)
        print ("Std:", np.std(episodes_to_solve),file=f)
        print ("Max:", np.max(episodes_to_solve),file=f)
        print ("Min:", np.min(episodes_to_solve),file=f)
        
        print (">> Rewards:",file=f)
        print ("Mean:",file=f)
        print(mean_rewards,file=f)
        print ("Std:",file=f)
        print (std_rewards,file=f)
            
        print()
        
    return [episodes_to_solve,mean_rewards,std_rewards]

# Trials DQN
[episodes_to_solve,mean_rewards,std_rewards] = run_dqn_trials()
plt.errorbar(range(1,NUM_EPISODES+1),mean_rewards,yerr=std_rewards,label="DQN")

# Trials DQN with evoReward
#n_population = 2
#[episodes_to_solve,mean_rewards,std_rewards] = run_dqn_trials(dqn_type=EVODQN_TYPE,num_population=n_population)
#plt.errorbar(range(1,NUM_EPISODES+1),mean_rewards,yerr=std_rewards,label="DQN-evoReward, pop: "+str(n_population))

plt.xlabel("Episode")
plt.ylabel("Average Cumulative Reward")
plt.legend(title="DQN type")
plt.savefig("result.png")