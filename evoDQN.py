import evoReward
import numpy as np
from collections import deque
import datetime
import time
import gym
import gym_minigrid

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

CROSS_RATE = 0.4  # mating probability (DNA crossover)
MUTATION_RATE = 0.01  # mutation probability
REWARD_BOUND = [-5, 5]  # Bounds for the augmented reward
EPSILON = 0.3
PRINT_INTERVAL = 1
LOGGING_MEAN_SIZE = 5
REWARD_MULTIPLIER = 100
SOLVE_SCORE = 0.90


class QNetwork:
    def __init__(self,
                 learning_rate=0.01,
                 state_size=4,
                 action_size=2,
                 hidden_size=10):
        # state inputs to the Q-network
        self.model = Sequential()

        self.model.add(
            Dense(hidden_size, activation='relu', input_dim=state_size))
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
        idx = np.random.choice(
            np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]


train_episodes = 1000  # max number of episodes to learn from
max_steps = 200  # max steps in an episode
gamma = 0.99  # future reward discount

# Exploration parameters
explore_start = 0.8  # exploration probability at start
explore_stop = 0.00  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob

# Network parameters
hidden_size = 16  # number of units in each Q-network hidden layer
learning_rate = 0.001  # Q-network learning rate

# Memory parameters
memory_size = 10000  # memory capacity
batch_size = 32  # experience mini-batch size
pretrain_length = batch_size  # number experiences to pretrain the memory

class DQNBase:
    def __init__(self,
                 env_wrapper=None,
                 score_averaged_over=50,
                 log_file='logs/log'):
        if env_wrapper is None:
            self.env = gym.make("MiniGrid-Empty-6x6-v0")
            self.obs_space_size = self.env.observation_space.spaces['image'].shape[0]**2
            self.action_space_size = self.env.action_space.n
            self.preprocess = lambda state: state['image'][:, :, 0].reshape(1, self.obs_space_size)
        else:
            self.env = env_wrapper.env
            self.obs_space_size = env_wrapper.obs_space_size
            self.action_space_size = env_wrapper.action_space_size
            self.preprocess = env_wrapper.preprocess

        self.log_file = log_file
        self.score_averaged_over = score_averaged_over

class DQNTrain(DQNBase):
    def __init__(self,
                 env_wrapper=None,
                 score_averaged_over=50,
                 log_file='logs/log'):
        
        super(DQNTrain, self).__init__(env_wrapper,score_averaged_over,log_file)

        # init DQN agent
        self.dqn_agent = QNetwork(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            state_size=self.obs_space_size,
            action_size=self.action_space_size)

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
                state, reward, done, _ = self.env.step(
                    self.env.action_space.sample())
                state = self.preprocess(state)
            else:
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))
                state = next_state

        return state

    def model_train(self):
        # Replay
        inputs = np.zeros((batch_size, self.obs_space_size))
        targets = np.zeros((batch_size, self.action_space_size))
        minibatch = self.memory.sample(batch_size)
        for i, (state_b, action_b, reward_b,
                next_state_b) in enumerate(minibatch):
            inputs[i:i + 1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(
                    state_b.shape)).all(axis=1):
                target = reward_b + gamma * np.amax(
                    self.dqn_agent.model.predict(next_state_b)[0])
            targets[i] = self.dqn_agent.model.predict(state_b)
            targets[i][action_b] = target
        self.dqn_agent.model.fit(inputs, targets, epochs=1, verbose=0)

    def train(self, n_episodes=1000, render=False, epsilon=EPSILON):
        state = self.populate_memory()

        #max steps
        n_steps = self.env.max_steps

        scores_deque = deque(maxlen=self.score_averaged_over)
        scores_deque_logging = deque(maxlen=LOGGING_MEAN_SIZE)
        scores = []

        for ep in range(1, n_episodes + 1):
            time_start = time.time()
            episode_reward = 0.0
            # Explore or Exploit
            explore_p = epsilon
            for t in range(1, n_steps + 1):
                self.env.render()

                if explore_p > np.random.rand():
                    # Make a random action
                    action = self.env.action_space.sample()
                else:
                    # Get action from Q-network
                    Qs = self.dqn_agent.model.predict(state)[0]
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
                    state, reward, done, _ = self.env.step(
                        self.env.action_space.sample())
                    state = self.preprocess(state)
                    break
                else:
                    # Add experience to memory
                    self.memory.add((state, action, reward, next_state))
                    state = next_state

                self.model_train()

            scores_deque.append(episode_reward)
            scores_deque_logging.append(episode_reward)
            scores.append(episode_reward)

            # find time taken for each trial
            time_end = time.time()
            time_taken = (time_end - time_start) / 60

            if (ep % PRINT_INTERVAL == 0):
                print(
                    'Episode {}\tAverage Score({:d}): {:.3f}\tCurrent Score: {:.3f}\tTime: {:.2f}'.
                    format(ep, self.score_averaged_over, np.mean(scores_deque),
                           scores_deque[-1],time_taken))

        self.dqn_agent.model.save("models/dqn-reward-{}.model".format(
            datetime.datetime.now().strftime("%d-%m-%y %H:%M")))

        # render successful model
        if (render):
            state = self.preprocess(self.env.reset())
            is_done = False
            while (not is_done):
                self.env.render()
                action = self.dqn_agent.model.predict(state)[0]
                next_state, reward, is_done, _ = self.env.step(action)
                state = self.preprocess(next_state)

        return [np.array(scores)]


class evoAgent():
    def __init__(self, env_wrapper, averaged_over, id):
        self.env = env_wrapper.env
        self.obs_space_size = env_wrapper.obs_space_size
        self.action_space_size = env_wrapper.action_space_size
        self.preprocess = env_wrapper.preprocess

        self.qnetwork = QNetwork(
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            state_size=self.obs_space_size,
            action_size=self.action_space_size)
        self.memory = Memory(max_size=memory_size)
        self.state = self.populate_memory()
        self.last_mean_reward = 0.0
        self.scores_deque = deque(maxlen=averaged_over)
        self.explore_p = explore_start
        self.id = id

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
                state, reward, done, _ = self.env.step(
                    self.env.action_space.sample())
                state = self.preprocess(state)
            else:
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))
                state = next_state

        return state

    def model_train(self):
        # Replay
        inputs = np.zeros((batch_size, self.obs_space_size))
        targets = np.zeros((batch_size, self.action_space_size))

        minibatch = self.memory.sample(batch_size)
        for i, (state_b, action_b, reward_b,
                next_state_b) in enumerate(minibatch):
            inputs[i:i + 1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target = reward_b + gamma * np.amax(
                    self.qnetwork.model.predict(next_state_b)[0])
            targets[i] = self.qnetwork.model.predict(state_b)
            targets[i][action_b] = target
        self.qnetwork.model.fit(inputs, targets, epochs=1, verbose=0)


class EvoDQNTrain(DQNBase):
    def __init__(self,
                 env_wrapper,
                 score_averaged_over=50,
                 log_file='logs/log',
                 cross_rate=CROSS_RATE,
                 mutation_rate=MUTATION_RATE,
                 reward_bound=REWARD_BOUND):

        super(EvoDQNTrain, self).__init__(env_wrapper,score_averaged_over,log_file)
        self.env_wrapper = env_wrapper
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.reward_bound = reward_bound

    def train(self,
              n_episodes=1000,
              n_population=10,
              render=False,
              epsilon=EPSILON):
        with open(self.log_file, 'a+') as f:
            print(file=f)
            print("--evoDQN stats--", file=f)
            print("ENVIRONMENT:", self.env_wrapper.env_name, file=f)
            print("AVERAGED_OVER:", self.score_averaged_over, file=f)
            print("NUM_EPISODES:", n_episodes, file=f)
            print()

        # Init augmented reward object
        evo_reward_object = evoReward.evoReward(
            DNA_bound=self.reward_bound,
            cross_rate=self.cross_rate,
            mutation_rate=self.mutation_rate,
            pop_size=n_population,
            obs_space_size=self.obs_space_size,
            action_size=self.action_space_size,
            dna_type=1)

        # Init population of agents
        dqn_agents = []
        for n in range(n_population):
            dqn_agents.append(
                evoAgent(self.env_wrapper, self.score_averaged_over, n))

        #max steps
        n_steps = dqn_agents[0].env.max_steps

        best_agent_scores = []
        best_agent_scores_dequeue = deque(maxlen=self.score_averaged_over)
        scores = []
        scores_deque = deque(maxlen=self.score_averaged_over)
        augmented_reward_max = None
        model_max = None
        agent_mean = np.zeros(n_population)
        for ep in range(1, n_episodes + 1):
            episode_reward_population = np.array([])
            time_start = time.time()

            for n in range(n_population):
                episode_reward = 0.0
                agent = dqn_agents[n]
                agent.explore_p = epsilon

                for t in range(1, n_steps + 1):
                    # Explore or Exploit
                    if agent.explore_p > np.random.rand():
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

                    # augment reward
                    reward = reward * REWARD_MULTIPLIER
                    reward += evo_reward_object.get_reward(
                        n, action, agent.state.flatten())

                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros(agent.state.shape)

                        # Add experience to memory
                        agent.memory.add((agent.state, action, reward,
                                          next_state))

                        # Start new episode
                        agent.env.reset()
                        # Take one random step to get the pole and cart moving
                        agent.state, reward, done, _ = agent.env.step(
                            agent.env.action_space.sample())
                        agent.state = self.preprocess(agent.state)
                        break
                    else:
                        # Add experience to memory
                        agent.memory.add((agent.state, action, reward,
                                          next_state))
                        agent.state = next_state

                    agent.model_train()

                episode_reward_population = np.append(
                    episode_reward_population, episode_reward)
                agent.scores_deque.append(episode_reward)
                agent_mean[n] = np.mean(agent.scores_deque)

            # Add best agent rewards to list
            best_index = np.argmax(agent_mean)
            best_agent_score = episode_reward_population[best_index]
            best_agent_scores.append(best_agent_score)
            best_agent_scores_dequeue.append(best_agent_score)

            # Add max reward to list
            max_index = np.argmax(episode_reward_population)
            episode_reward_max = episode_reward_population[max_index]
            scores_deque.append(episode_reward_max)
            scores.append(episode_reward_max)

            # find time taken for each episode
            time_end = time.time()
            time_taken = (time_end - time_start) / 60

            if (ep % PRINT_INTERVAL == 0):
                with open(self.log_file, 'a+') as f:
                    print(
                        'Episode {}\tAvgMax({:d}): {:.2f}\tCurrentMax: {:.2f}\tAvgBestAgent({:d}): {:.2f}\tCurrentBestAgent: {:.2f}\tTime: {:.2f}'.
                        format(ep, self.score_averaged_over,
                               np.mean(scores_deque), episode_reward_max,
                               self.score_averaged_over,
                               np.mean(best_agent_scores_dequeue),
                               best_agent_score, time_taken),
                        file=f)
            if ep == n_episodes:
                model_max = dqn_agents[best_index]
                #model_max.qnetwork.model.save("models/grid_evodqn-population{}-{}-{}.model".format(n_population,self.env_name, datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")))
                augmented_reward_max = evo_reward_object.get_DNA(best_index)
                break

            # Evolve rewards
            evo_reward_object.set_fitness(agent_mean)
            evo_reward_object.evolve()

        # render successful model
        if (render):
            state = self.preprocess(model_max.env.reset())
            is_done = False
            while (not is_done):
                model_max.env.render()
                action = model_max.qnetwork.model.predict(state)[0]
                next_state, reward, is_done, _ = model_max.env.step(action)
                state = self.preprocess(next_state)

        return [
            np.array(scores),
            np.array(best_agent_scores), augmented_reward_max
        ]

class EvoDQNTrainParallel(DQNBase):
    def __init__(self,
                 env_wrapper,
                 score_averaged_over=50,
                 log_file='logs/log',
                 cross_rate=CROSS_RATE,
                 mutation_rate=MUTATION_RATE,
                 reward_bound=REWARD_BOUND):

        super(EvoDQNTrainParallel, self).__init__(env_wrapper,score_averaged_over,log_file)
        self.env_wrapper = env_wrapper
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.reward_bound = reward_bound

    def train(self,
              n_episodes=1000,
              n_population=8,
              render=False,
              epsilon=EPSILON):
        with open(self.log_file, 'a+') as f:
            print(file=f)
            print("--evoDQN stats--", file=f)
            print("ENVIRONMENT:", self.env_wrapper.env_name, file=f)
            print("AVERAGED_OVER:", self.score_averaged_over, file=f)
            print("NUM_EPISODES:", n_episodes, file=f)
            print()

        # Init augmented reward object
        evo_reward_object = evoReward.evoReward(
            DNA_bound=self.reward_bound,
            cross_rate=self.cross_rate,
            mutation_rate=self.mutation_rate,
            pop_size=n_population,
            obs_space_size=self.obs_space_size,
            action_size=self.action_space_size,
            dna_type=1)

        # Init population of agents
        dqn_agents = []
        for n in range(n_population):
            dqn_agents.append(
                evoAgent(self.env_wrapper, self.score_averaged_over, n))

        #max steps
        n_steps = dqn_agents[0].env.max_steps

        best_agent_scores = []
        best_agent_scores_dequeue = deque(maxlen=self.score_averaged_over)
        scores = []
        scores_deque = deque(maxlen=self.score_averaged_over)
        augmented_reward_max = None
        model_max = None
        agent_mean = np.zeros(n_population)
        for ep in range(1, n_episodes + 1):
            episode_reward_population = np.array([])
            time_start = time.time()

            for n in range(n_population):
                episode_reward = 0.0
                agent = dqn_agents[n]
                agent.explore_p = epsilon

                for t in range(1, n_steps + 1):
                    # Explore or Exploit
                    if agent.explore_p > np.random.rand():
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

                    # augment reward
                    reward = reward * REWARD_MULTIPLIER
                    reward += evo_reward_object.get_reward(
                        n, action, agent.state.flatten())

                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros(agent.state.shape)

                        # Add experience to memory
                        agent.memory.add((agent.state, action, reward,
                                          next_state))

                        # Start new episode
                        agent.env.reset()
                        # Take one random step to get the pole and cart moving
                        agent.state, reward, done, _ = agent.env.step(
                            agent.env.action_space.sample())
                        agent.state = self.preprocess(agent.state)
                        break
                    else:
                        # Add experience to memory
                        agent.memory.add((agent.state, action, reward,
                                          next_state))
                        agent.state = next_state

                    agent.model_train()

                episode_reward_population = np.append(
                    episode_reward_population, episode_reward)
                agent.scores_deque.append(episode_reward)
                agent_mean[n] = np.mean(agent.scores_deque)

            # Add best agent rewards to list
            best_index = np.argmax(agent_mean)
            best_agent_score = episode_reward_population[best_index]
            best_agent_scores.append(best_agent_score)
            best_agent_scores_dequeue.append(best_agent_score)

            # Add max reward to list
            max_index = np.argmax(episode_reward_population)
            episode_reward_max = episode_reward_population[max_index]
            scores_deque.append(episode_reward_max)
            scores.append(episode_reward_max)

            # find time taken for each episode
            time_end = time.time()
            time_taken = (time_end - time_start) / 60

            if (ep % PRINT_INTERVAL == 0):
                with open(self.log_file, 'a+') as f:
                    print(
                        'Episode {}\tAvgMax({:d}): {:.2f}\tCurrentMax: {:.2f}\tAvgBestAgent({:d}): {:.2f}\tCurrentBestAgent: {:.2f}\tTime: {:.2f}'.
                        format(ep, self.score_averaged_over,
                               np.mean(scores_deque), episode_reward_max,
                               self.score_averaged_over,
                               np.mean(best_agent_scores_dequeue),
                               best_agent_score, time_taken),
                        file=f)
            if ep == n_episodes:
                model_max = dqn_agents[best_index]
                #model_max.qnetwork.model.save("models/grid_evodqn-population{}-{}-{}.model".format(n_population,self.env_name, datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")))
                augmented_reward_max = evo_reward_object.get_DNA(best_index)
                break

            # Evolve rewards
            evo_reward_object.set_fitness(agent_mean)
            evo_reward_object.evolve()

        # render successful model
        if (render):
            state = self.preprocess(model_max.env.reset())
            is_done = False
            while (not is_done):
                model_max.env.render()
                action = model_max.qnetwork.model.predict(state)[0]
                next_state, reward, is_done, _ = model_max.env.step(action)
                state = self.preprocess(next_state)

        return [
            np.array(scores),
            np.array(best_agent_scores), augmented_reward_max
        ]

    @staticmethod
    def run_agent_episode(agent, preprocess, evo_reward_object, epsilon, n_steps):
        episode_reward = 0.0
        agent.explore_p = epsilon

        for t in range(1, n_steps + 1):
            # Explore or Exploit
            if agent.explore_p > np.random.rand():
                # Make a random action
                action = agent.env.action_space.sample()
            else:
                # Get action from Q-network
                Qs = agent.qnetwork.model.predict(agent.state)[0]
                action = np.argmax(Qs)

            # step environment
            next_state, reward, done, _ = agent.env.step(action)
            next_state = preprocess(next_state)
            episode_reward += reward

            # augment reward
            reward = reward * REWARD_MULTIPLIER
            reward += evo_reward_object.get_reward(
                agent.id, action, agent.state.flatten())

            if done:
                # the episode ends so no next state
                next_state = np.zeros(agent.state.shape)

                # Add experience to memory
                agent.memory.add((agent.state, action, reward,
                                  next_state))

                # Start new episode
                agent.env.reset()
                # Take one random step to get the pole and cart moving
                agent.state, reward, done, _ = agent.env.step(
                    agent.env.action_space.sample())
                agent.state = preprocess(agent.state)
                break
            else:
                # Add experience to memory
                agent.memory.add((agent.state, action, reward,
                                  next_state))
                agent.state = next_state

            agent.model_train()

