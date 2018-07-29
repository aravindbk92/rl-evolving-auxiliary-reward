import gym
import evoReward
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

CROSS_RATE = 0.4                    # mating probability (DNA crossover)
MUTATION_RATE = 0.01                # mutation probability
REWARD_BOUND = [-10,10]               # Bounds for the augmented reward
PRINT_INTERVAL = 10

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10):
        # state inputs to the Q-network
        self.model = Sequential()

        self.model.add(Dense(hidden_size, activation='relu',
                             input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]

train_episodes = 1000          # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 16               # number of units in each Q-network hidden layer
learning_rate = 0.001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 32                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

class DQNTrain:
    def __init__(self,env="CartPole-v0",solve_score=195.0):
        self.env = gym.make(env)
        self.solve_score = solve_score

    ## Populate the experience memory           
    def populate_memory(self):    
        # Initialize the simulation
        self.env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = self.env.step(self.env.action_space.sample())
        state = self.preprocess(state)
        
        self.memory = Memory(max_size=memory_size)
        
        # Make a bunch of random actions and store the experiences
        for ii in range(pretrain_length):
            # Make a random action
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.preprocess(next_state)
        
            if done:
                # The simulation fails so no next state
                next_state = np.zeros(state.shape)
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))
        
                # Start new episode
                self.env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = self.env.step(self.env.action_space.sample())
                state = self.preprocess(state)
            else:
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))
                state = next_state
        
        return state
    
    def preprocess(self, state):
        return state.reshape(1,self.env.observation_space.shape[0])
    
    def train(self, n_episodes=1000,n_steps=200, render=False):
        state = self.populate_memory()
        
        # init DQN agent
        dqn_agent = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)

        scores_deque = deque(maxlen=100)
        scores = []
        step = 0
        for ep in range(1,n_episodes+1):
            episode_reward = 0.0
            for t in range(1,n_steps+1):
                step += 1
                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = self.env.action_space.sample()
                else:
                    # Get action from Q-network
                    Qs = dqn_agent.model.predict(state)[0]
                    action = np.argmax(Qs)
                    
                # step environment
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess(next_state)
                episode_reward += reward
                
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
        
                    # Add experience to memory
                    self.memory.add((state, action, reward, next_state))
        
                    # Start new episode
                    self.env.reset()
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = self.env.step(self.env.action_space.sample())
                    state = self.preprocess(state)
                    break
                else:
                    # Add experience to memory
                    self.memory.add((state, action, reward, next_state))
                    state = next_state

                # Replay
                inputs = np.zeros((batch_size, 4))
                targets = np.zeros((batch_size, 2))                    
                
                minibatch = self.memory.sample(batch_size)
                for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                    inputs[i:i+1] = state_b
                    target = reward_b
                    if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                        target = reward_b + gamma * np.amax(dqn_agent.model.predict(next_state_b)[0])
                    targets[i] = dqn_agent.model.predict(state_b)
                    targets[i][action_b] = target
                dqn_agent.model.fit(inputs, targets, epochs=1, verbose=0)
                
            scores_deque.append(episode_reward)
            scores.append(episode_reward) 
            if (ep % PRINT_INTERVAL == 0):
                print('Episode {}\tAverage Score: {:.2f}\tCurrent Score: {:.2f}'.format(ep, np.mean(scores_deque),scores_deque[-1]))
            if np.mean(scores_deque)>=self.solve_score:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep-100, np.mean(scores_deque)))
                break

        dqn_agent.model.save("models/dqn-reward{}.model".format(scores_deque[-1]))

        # render successful model
        if (render):
            state = self.preprocess(self.env.reset())
            is_done = False
            while(not is_done):
                self.env.render()
                action = dqn_agent.model.predict(state)[0]
                next_state, reward, is_done, _ = self.env.step(action)
                state = self.preprocess(next_state)

        return [np.array(scores)]

class evoAgent:
    def __init__(self,env):
        self.env = gym.make(env)
        self.obs_space_size = self.env.observation_space.shape[0]
        self.qnetwork = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)
        self.memory = Memory(max_size=memory_size)
        self.state = self.populate_memory()
        self.step = 0
        
    def preprocess(self, state):
        return state.reshape(1,self.obs_space_size)        
        
    def populate_memory(self):
        # Initialize the simulation
        self.env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = self.env.step(self.env.action_space.sample())
        state = self.preprocess(state)
        
        # Make a bunch of random actions and store the experiences
        for ii in range(pretrain_length):
            # Make a random action
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.preprocess(next_state)
        
            if done:
                # The simulation fails so no next state
                next_state = np.zeros(state.shape)
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))
        
                # Start new episode
                self.env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = self.env.step(self.env.action_space.sample())
                state = self.preprocess(state)
            else:
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))
                state = next_state
        
        return state
    
    def model_train(self):
        # Replay
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))                    
        
        minibatch = self.memory.sample(batch_size)
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
            inputs[i:i+1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target = reward_b + gamma * np.amax(self.qnetwork.model.predict(next_state_b)[0])
            targets[i] = self.qnetwork.model.predict(state_b)
            targets[i][action_b] = target
        self.qnetwork.model.fit(inputs, targets, epochs=1, verbose=0)        

class EvoDQNTrain:
    def __init__(self,env="CartPole-v0",solve_score=195.0,cross_rate=CROSS_RATE,mutation_rate=MUTATION_RATE,reward_bound=REWARD_BOUND):
        self.solve_score = solve_score
        self.env_name = env
        
        env_temp = gym.make(env)
        self.obs_space_size = env_temp.observation_space.shape[0]
        self.action_space_size = env_temp.action_space.n 
        
        self.cross_rate=cross_rate
        self.mutation_rate=mutation_rate
        self.reward_bound=reward_bound
   
    def preprocess(self, state):
        return state.reshape(1,self.obs_space_size)
    
    def train(self, n_episodes=1000,n_steps=200, n_population=2, render=False):
        self.n_population = n_population
        
        # Init augmented reward object
        self.evoRewardObject = evoReward.evoReward(DNA_bound=self.reward_bound, cross_rate=self.cross_rate,mutation_rate=self.mutation_rate, pop_size=self.n_population,obs_space_size=self.obs_space_size,action_size=self.action_space_size,dna_type=1)

        # Init population of agents
        dqn_agents = []
        for n in range(self.n_population):
             dqn_agents.append(evoAgent(self.env_name))

        scores_deque = deque(maxlen=100)
        scores = []
        augmented_reward_max = None
        model_max = None
        for ep in range(1,n_episodes+1):
            episode_reward_population = np.array([])
            
            for n in range(n_population):
                episode_reward = 0.0
                agent = dqn_agents[n]
                for t in range(1,n_steps+1):
                    agent.step += 1
                    
                    # Explore or Exploit
                    explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*agent.step)
                    if explore_p > np.random.rand():
                        # Make a random action
                        action = agent.env.action_space.sample()
                    else:
                        # Get action from Q-network
                        Qs = agent.qnetwork.model.predict(agent.state)[0]
                        action = np.argmax(Qs)
                        
                    # step environment
                    next_state, reward, done, _ = agent.env.step(action)
                    next_state = self.preprocess(next_state)
                    episode_reward += reward
                    
                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros(agent.state.shape)
            
                        # Add experience to memory
                        agent.memory.add((agent.state, action, reward, next_state))
            
                        # Start new episode
                        agent.env.reset()
                        # Take one random step to get the pole and cart moving
                        agent.state, reward, done, _ = agent.env.step(agent.env.action_space.sample())
                        agent.state = self.preprocess(agent.state)
                        break
                    else:
                        # Add experience to memory
                        agent.memory.add((agent.state, action, reward, next_state))
                        agent.state = next_state
    
                    agent.model_train()
                    
                episode_reward_population = np.append(episode_reward_population,episode_reward)

            # Add best reward to list
            max_index = np.argmax(episode_reward_population)
            episode_reward_max = episode_reward_population[max_index]                
            scores_deque.append(episode_reward_max)            
            scores.append(episode_reward_max) 
            
            if (ep % PRINT_INTERVAL == 0):
                print('Episode {}\tAverage Score: {:.2f}\tCurrent Score: {:.2f}'.format(ep, np.mean(scores_deque),scores_deque[-1]))
            if np.mean(scores_deque)>=self.solve_score:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep-100, np.mean(scores_deque)))
                model_max = dqn_agents[max_index]
                model_max.qnetwork.model.save("models/evodqn-reward{}.model".format(scores_deque[-1]))
                augmented_reward_max = self.evoRewardObject.get_DNA(max_index)
                break
            
            # Evolve rewards
            #self.evoRewardObject.set_fitness(np.array(episode_reward_population))
            #self.evoRewardObject.evolve()

        # render successful model
        if (render):
            state = self.preprocess(model_max.env.reset())
            is_done = False
            while(not is_done):
                model_max.env.render()
                action = model_max.qnetwork.model.predict(state)[0]
                next_state, reward, is_done, _ = model_max.env.step(action)
                state = self.preprocess(next_state)

        return [np.array(scores),augmented_reward_max] 