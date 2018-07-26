import numpy as np

class RL:
    def __init__(self,mdp,sampleReward,instanceID=0,evoRewardObject=None):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward
        self.instanceID = instanceID
        self.evoRewardObject = evoRewardObject
        
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
    
    def getAugmentedReward(self,state, action):
        if self.evoRewardObject is None:
            return 0
        else:
            return self.evoRewardObject.getReward(self.instanceID, action, state, self.mdp.nStates)

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0, evoReward=False):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''
        Q_old = initialQ
        Q = initialQ
        
        visit_count = np.zeros((self.mdp.nActions,self.mdp.nStates))
        episode_rewards = []
        for n_episodes in range(nEpisodes):
            episode_rewards.append(0.0)
            s = s0
            for n_steps in range(nSteps):
                if np.random.rand() < epsilon:
                    a = np.random.randint(0,self.mdp.nActions)
                else:
                    if (temperature > 0):
                        boltzmann_prob = np.exp(Q[:,s]/temperature)
                        boltzmann_prob = boltzmann_prob/np.sum(boltzmann_prob)
                        a = np.argmax(boltzmann_prob)
                    else:
                        a = np.argmax(Q[:,s])
                    
                [r,s_prime] = self.sampleRewardAndNextState(s,a)
                episode_rewards[-1] += r
                
                if (evoReward):
                    r += self.getAugmentedReward(s,a)
                visit_count[a,s] += 1
                alpha = 1.0/visit_count[a,s]
                Q[a,s] = Q_old[a,s] + alpha * (r + self.mdp.discount * np.max(Q_old[:,s_prime]) - Q_old[a,s])
                Q_old = Q
                s = s_prime
        
        policy = np.argmax(Q,axis=0)
        return [Q,policy,np.array(episode_rewards)]