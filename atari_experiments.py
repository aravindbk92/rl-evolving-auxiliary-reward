import matplotlib.pyplot as plt
from atari import DQNTrain, EvoDQNTrain
import numpy as np
import time

NUM_TRIALS = 2
NUM_EPISODES = 1000
SOLVED_REWARD_CRITERIA = 195.0
AVERAGED_OVER = 50

DQN_TYPE = 0
EVODQN_TYPE = 1

def run_dqn_trials(dqn_type=DQN_TYPE, num_population=30):
    episode_reward_trials = np.empty((0,NUM_EPISODES))
    episodes_to_solve = np.array([])
    
    dqnobject=None
    if dqn_type == EVODQN_TYPE:
        print ("----DQN with augmented reward----")
        dqnobject = EvoDQNTrain(env="CartPole-v0", solve_score=SOLVED_REWARD_CRITERIA)
    else:
        print ("----DQN with normal reward----")
        dqnobject = DQNTrain(env="CartPole-v0", solve_score=SOLVED_REWARD_CRITERIA)
    
    max_num_episodes_trial = 0
    min_num_episodes_trial = NUM_EPISODES
    best_augment = None
    time_trials = []
    for trial in range(NUM_TRIALS):
        print()
        print ("Trial:", trial)
        time_start = time.time()        
        
        if dqn_type == EVODQN_TYPE:
            [episode_rewards,best_agent_rewards,augment] = dqnobject.train(n_episodes=NUM_EPISODES,n_population=num_population)
        else:
            [episode_rewards] = dqnobject.train(n_episodes=NUM_EPISODES)
        
        # find in which episode it was solved for current trial
        episodes_to_solve = np.append(episodes_to_solve,episode_rewards.size)

        # find maximum number of episodes reached across trials
        if (episode_rewards.size > max_num_episodes_trial):
            max_num_episodes_trial = episode_rewards.size
                
        # find minimum number of episodes reached across trials
        if (episode_rewards.size < min_num_episodes_trial):
            min_num_episodes_trial = episode_rewards.size
            if dqn_type == EVODQN_TYPE:
                best_augment = augment
                
        # append reward history from each trial
        episode_rewards = np.pad(episode_rewards,(0,NUM_EPISODES-episode_rewards.size),mode='constant',constant_values=SOLVED_REWARD_CRITERIA)
        episode_reward_trials = np.append(episode_reward_trials,[episode_rewards],axis=0)
        
        # find time taken for each trial
        time_end = time.time()
        time_taken = (time_end-time_start)/60
        time_trials.append(time_taken)
        print ("------Trial: ", trial, " took ", time_taken, " minutes--------")
    
    # find mean and std
    episode_reward_trials = episode_reward_trials[:,:max_num_episodes_trial]
    mean_rewards = np.average(episode_reward_trials,axis=0)    
    std_rewards = np.std(episode_reward_trials,axis=0)
    
    # print stats
    file_name = ('logs/log_evoreward_pop'+str(num_population) if dqn_type == EVODQN_TYPE else 'log_dqn')
    file_name += "_"+str(NUM_TRIALS)+"_trials"
    with open(file_name, 'w') as f:
        print (file=f)
        print ("--DQN stats--",file=f) if (dqn_type==DQN_TYPE) else print ("--DQN with evoRewards stats--",file=f)
        print (">> Episodes taken to solve (reach score: 195.0):",file=f)
        print ("Mean:", np.average(episodes_to_solve)-AVERAGED_OVER,file=f)
        print ("Std:", np.std(episodes_to_solve),file=f)
        print ("Max:", max_num_episodes_trial-AVERAGED_OVER,file=f)
        print ("Min:", min_num_episodes_trial-AVERAGED_OVER,file=f)
        
        print (file=f)
        print (">> Rewards:",file=f)
        print ("Mean:",file=f)
        print(mean_rewards,file=f)
        print ("Std:",file=f)
        print (std_rewards,file=f)
        
        if dqn_type == EVODQN_TYPE:
            print ("Best augment:",file=f)
            print (best_augment,file=f)
            
        print (file=f)
        print (">> Time:",file=f)
        for index,time_taken in enumerate(time_trials):
            print ("Trial ",(index+1), ": ",time_taken,file=f)
        print (">>Total time taken: ", sum(time_trials), file=f)
        print()
        
    return [episodes_to_solve,mean_rewards,std_rewards]

# Trials DQN with evoReward
n_population = 10
[episodes_to_solve,mean_rewards,std_rewards] = run_dqn_trials(dqn_type=EVODQN_TYPE, num_population=n_population)
plt.errorbar(range(1,mean_rewards.size+1),mean_rewards,yerr=std_rewards,label="DQN-evoReward, pop: "+str(n_population))

# Trials DQN
[episodes_to_solve,mean_rewards,std_rewards] = run_dqn_trials()
plt.errorbar(range(1,mean_rewards.size+1),mean_rewards,yerr=std_rewards,label="DQN")

plt.xlabel("Episode")
plt.ylabel("Average Cumulative Reward")
plt.legend(title="DQN type")
plt.savefig("result.png")