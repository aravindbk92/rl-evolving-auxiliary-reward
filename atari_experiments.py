import matplotlib.pyplot as plt
from atari import AtariTrain
import numpy as np
import math

NUM_TRIALS = 100
NUM_EPISODES = 100
SOLVED_REWARD_CRITERIA = 195.0

DQN_TYPE = 0
EVODQN_TYPE = 1

atari = AtariTrain(env="CartPole-v0")

def run_dqn_trials(dqn_type=DQN_TYPE, num_population=40):
    episode_reward_trials = np.empty((0,NUM_EPISODES))
    episodes_to_solve = np.array([])
    if dqn_type == EVODQN_TYPE:
        print ("----DQN with augmented reward----")    
    else:
        print ("----DQN with normal reward----")
    for trial in range(NUM_TRIALS):
        print()
        print ("trial:", trial)
        
        if dqn_type == EVODQN_TYPE:
            [episode_rewards,augment] = atari.evodqn_train(n_episodes=NUM_EPISODES, n_population=num_population)
        else:
            [episode_rewards] = atari.dqn_train(n_episodes=NUM_EPISODES)
        
        # find in which episode it was solved for current trial
        solved_episode = np.argmax(episode_rewards>=SOLVED_REWARD_CRITERIA)
        if (episode_rewards[solved_episode] >= SOLVED_REWARD_CRITERIA):
            solved_episode += 1
        else:
            solved_episode = NUM_EPISODES
        episodes_to_solve = np.append(episodes_to_solve,solved_episode)
        
        # append reward history from each trial
        episode_reward_trials = np.append(episode_reward_trials,[episode_rewards],axis=0)
    
    # find mean and std
    mean_rewards = np.average(episode_reward_trials,axis=0)    
    std_rewards = np.std(episode_reward_trials,axis=0)
    
    # print stats
    print ()
    print ("--DQN stats--") if (dqn_type==DQN_TYPE) else print ("--DQN with evoRewards stats--")
    print (">> Episodes taken to solve (reach score: 195.0):")
    print ("Mean:", np.average(episodes_to_solve))
    print ("Std:", np.std(episodes_to_solve))
    print ("Max:", np.max(episodes_to_solve))
    print ("Min:", np.min(episodes_to_solve))
    
    solved_episode_index = np.argmax(mean_rewards>=SOLVED_REWARD_CRITERIA)    
    if (mean_rewards[solved_episode_index] >= SOLVED_REWARD_CRITERIA):
        print ("Solved in ",solved_episode_index+1, " episodes")
        
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