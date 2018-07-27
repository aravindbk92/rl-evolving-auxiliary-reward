import numpy as np
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
        
    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]
    
    def getAugmentedReward(self,state, action, instanceID):
        if self.evoRewardObject is None:
            return 0
        else:
            return self.evoRewardObject.get_reward(instanceID, action, state, self.mdp.nStates)        
        
    def qLearningAugmented(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0, n_population=10, cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE,reward_bound=REWARD_BOUND):
        # Instantiate reward GA object
        self.evoRewardObject = evoReward.evoReward(DNA_size=self.mdp.nStates*self.mdp.nActions, DNA_bound=reward_bound, cross_rate=cross_rate,mutation_rate=mutation_rate, pop_size=n_population)
        
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
            
            # Run one iteration for each agent
            for n in range(n_population):
                # Use initialQ is first iteration, previous Q if not
                if n_episodes == 0:
                    [Q,episode_reward] = self.QLearnEpisode(s0=s0,initialQ=initialQ,nSteps=nSteps,epsilon=epsilon, instanceID=n)
                else:
                    [Q,episode_reward] = self.QLearnEpisode(s0=s0,initialQ=Q_population[n],nSteps=nSteps,epsilon=epsilon, instanceID=n)
                
                # Set results over the whole population on one iteration
                Q_population[n] = Q
                episode_reward_population = np.append(episode_reward_population,episode_reward)
            
            # Add best reward to list
            max_index = np.argmax(episode_reward_population)
            episode_reward_max = np.append(episode_reward_max,episode_reward_population[max_index])
            
            # If last episode, find best Q model
            if (n_episodes == nEpisodes-1):                
                Q_max = Q_population[max_index]
                augmented_reward_max = self.evoRewardObject.get_DNA(max_index)
                break                
            
            # Evolve rewards
            self.evoRewardObject.set_fitness(np.array(episode_reward_population))
            self.evoRewardObject.evolve()
            
        policy_max = np.argmax(Q_max,axis=0)
        
        return [Q_max,policy_max,episode_reward_max,augmented_reward_max]
    
    def QLearnEpisode(self, s0,initialQ,nSteps,epsilon=0, instanceID=0):
        Q_old = initialQ
        Q = initialQ
        
        visit_count = np.zeros((self.mdp.nActions,self.mdp.nStates))
        episode_reward = 0.0
        s = s0
        for n_steps in range(nSteps):
            if np.random.rand() < epsilon:
                a = np.random.randint(0,self.mdp.nActions)
            else:
                    a = np.argmax(Q[:,s])
                
            [r,s_prime] = self.sampleRewardAndNextState(s,a)
            episode_reward += r
            
            r += self.getAugmentedReward(s,a,instanceID)
            
            visit_count[a,s] += 1
            alpha = 1.0/visit_count[a,s]
            Q[a,s] = Q_old[a,s] + alpha * (r + self.mdp.discount * np.max(Q_old[:,s_prime]) - Q_old[a,s])
            Q_old = Q
            s = s_prime

        return [Q,episode_reward]