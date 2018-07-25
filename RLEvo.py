import numpy as np
import MDP
import RL
import evoReward

class RLEvo:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward
        
    def qLearningAugmented(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0, initial_population=10):
        population = []
        evoRewardObject = evoReward.evoReward(initial_population)
        
        for n in range(initial_population):
            population.append(RL.RL(self.mdp,self.sampleReward,instanceID=n,evoRewardObject=evoRewardObject))
        Q_population = list(range(initial_population))
        
        episode_reward_max = np.array([])
        episode_reward_population = []
        Q_max = None
        policy_max = None
        for n_episodes in range(nEpisodes):
            episode_reward_population = []
            policy_population = []
            for n in range(population.size):                
                if n_episodes == 0:
                    [Q,policy, episode_rewards] = population[n].qLearning(s0=s0,initialQ=initialQ,nEpisodes=1,nSteps=nSteps,epsilon=epsilon,evoReward=True)
                else:
                    [Q,policy, episode_rewards] = population[n].qLearning(s0=s0,initialQ=Q_population[n],nEpisodes=1,nSteps=nSteps,epsilon=epsilon,evoReward=True)
                Q_population[n] = Q
                policy_population.append(policy)
                episode_reward_population.append(episode_rewards)
        
            evoRewardObject.evolve()

        max_index = np.argmax(episode_reward_population)
        np.append(episode_reward_max,episode_reward_population[max_index])
        Q_max = Q_population[max_index]
        policy_max = policy_population[max_index]
        
        return [Q_max,policy_max,episode_reward_max]