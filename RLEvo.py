import numpy as np
import RL
import evoReward

CROSS_RATE = 0.4                    # mating probability (DNA crossover)
MUTATION_RATE = 0.01                # mutation probability
REWARD_BOUND = [0,50]               # Bounds for the augmented reward

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
        
    def qLearningAugmented(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0, n_population=10, cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE,reward_bound=REWARD_BOUND):
        # Instantiate reward GA object
        evoRewardObject = evoReward.evoReward(DNA_size=self.mdp.nStates*self.mdp.nActions, DNA_bound=reward_bound, cross_rate=cross_rate,mutation_rate=mutation_rate, pop_size=n_population)
        
        # Initialize population of Q-agents
        agent_population = []
        for n in range(n_population):
            agent_population.append(RL.RL(self.mdp,self.sampleReward,instanceID=n,evoRewardObject=evoRewardObject))
            
        # Initialize variables for Algo
        Q_max = None
        policy_max = None
        augmented_reward_max = None
        episode_reward_max = np.array([])
        Q_population = list(range(n_population))
        # Run episodes
        for n_episodes in range(nEpisodes):
            # Episodic variables
            episode_reward_population = np.array([])
            policy_population = []            
            
            # Run one iteration for each agent
            for n in range(n_population):
                # Use initialQ is first iteration, previous Q if not
                if n_episodes == 0:
                    [Q,policy, episode_rewards] = agent_population[n].qLearning(s0=s0,initialQ=initialQ,nEpisodes=1,nSteps=nSteps,epsilon=epsilon,evoReward=True)
                else:
                    [Q,policy, episode_rewards] = agent_population[n].qLearning(s0=s0,initialQ=Q_population[n],nEpisodes=1,nSteps=nSteps,epsilon=epsilon,evoReward=True)
                
                # Set results over the whole population on one iteration
                Q_population[n] = Q
                policy_population.append(policy)
                episode_reward_population = np.append(episode_reward_population,episode_rewards[0])
            
            # Add best reward to list
            max_index = np.argmax(episode_reward_population)
            episode_reward_max = np.append(episode_reward_max,episode_reward_population[max_index])
            
            # If last episode, find best Q model
            if (n_episodes == nEpisodes-1):                
                Q_max = Q_population[max_index]
                policy_max = policy_population[max_index]
                augmented_reward_max = evoRewardObject.getDNA(max_index)
                break                
            
            # Evolve rewards
            evoRewardObject.set_fitness(np.array(episode_reward_population))
            evoRewardObject.evolve()
        
        return [Q_max,policy_max,episode_reward_max,augmented_reward_max]