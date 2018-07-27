import gym
import evoReward
from dqn import DQN
import numpy as np

CROSS_RATE = 0.4                    # mating probability (DNA crossover)
MUTATION_RATE = 0.01                # mutation probability
REWARD_BOUND = [-10,10]               # Bounds for the augmented reward

class AtariTrain:
    def __init__(self,env="CartPole-v0"):
        self.env = gym.make(env)

    def dqn_train(self, gamma=0.9,epsilon=0.95,n_episodes=100,n_steps=200, render=False):
        # init DQN agent
        dqn_agent = DQN(env=self.env,gamma=gamma,epsilon=epsilon)

        episode_reward_history = np.array([])
        for episode in range(n_episodes):
            cur_state = self.preprocess(self.env.reset())
            episode_reward = 0.0
            for step in range(n_steps):
                # step environment
                action = dqn_agent.act(cur_state)
                new_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                # train dqn
                new_state = self.preprocess(new_state)
                dqn_agent.remember(cur_state, action, reward, new_state, done)
                dqn_agent.replay()
                dqn_agent.target_train()

                # set next state as current state
                cur_state = new_state
                if done:
                    break
            episode_reward_history = np.append(episode_reward_history,episode_reward)
            print("Episode ",episode+1,"-> reward: ", episode_reward)

        dqn_agent.save_model("models/dqn-reward{}.model".format(episode_reward_history[-1]))

        # render successful model
        if (render):
            cur_state = self.preprocess(self.env.reset())
            is_done = False
            while(not is_done):
                self.env.render()
                action = dqn_agent.act(cur_state)
                new_state, reward, is_done, _ = self.env.step(action)
                cur_state = self.preprocess(new_state)

        return [episode_reward_history]

    def evodqn_train(self, gamma=0.9,epsilon=0.95,n_episodes=100,n_steps=200, render=False, n_population=10,cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE,reward_bound=REWARD_BOUND):
        # Init augmented reward object
        obs_space_size = self.env.observation_space.high.size
        action_space_size = self.env.action_space.n
        self.evoRewardObject = evoReward.evoReward(DNA_bound=reward_bound, cross_rate=cross_rate,mutation_rate=mutation_rate, pop_size=n_population,obs_space_size=obs_space_size,action_size=action_space_size,dna_type=1)

        # Init population of agents
        dqn_agents = []
        for n in range(n_population):
            dqn_agents.append(DQN(env=self.env,gamma=gamma,epsilon=epsilon))

        episode_reward_max_history = np.array([])
        model_max = None
        for episode in range(n_episodes):            
            episode_reward_population = np.array([])

            for n in range(n_population):
                cur_state = self.preprocess(self.env.reset())
                episode_reward = 0.0
                for step in range(n_steps):
                    action = dqn_agents[n].act(cur_state)
                    new_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward                    
                    
                    # augment reward
                    reward += self.evoRewardObject.get_reward(n, action, cur_state.flatten())
                    
                    #train DQN
                    new_state = self.preprocess(new_state)
                    dqn_agents[n].remember(cur_state, action, reward, new_state, done)
                    dqn_agents[n].replay()       # internally iterates default (prediction) model
                    dqn_agents[n].target_train() # iterates target model

                    # set new state as current
                    cur_state = new_state
                    if done:
                        break

                episode_reward_population = np.append(episode_reward_population,episode_reward)

            # Add best reward to list
            max_index = np.argmax(episode_reward_population)
            episode_reward_max = episode_reward_population[max_index]
            episode_reward_max_history = np.append(episode_reward_max_history,episode_reward_max)
            print("Episode ",episode+1,"-> max_reward: ", episode_reward_population[max_index])

            # If last episode, find best Q model
            if (episode == n_episodes-1):
                model_max = dqn_agents[max_index]
                augmented_reward_max = self.evoRewardObject.get_DNA(max_index)
                break

            # Evolve rewards
            self.evoRewardObject.set_fitness(np.array(episode_reward_population))
            self.evoRewardObject.evolve()

        model_max.save_model("models/evodqn-reward{}.model".format(episode_reward_max_history[-1]))

        # render successful model
        if (render):
            cur_state = self.preprocess(self.env.reset())
            is_done = False
            while(not is_done):
                self.env.render()
                action = model_max.act(cur_state)
                new_state, reward, is_done, _ = self.env.step(action)
                cur_state = self.preprocess(new_state)

        return [episode_reward_max_history,augmented_reward_max]

    def preprocess(self,state):
        return state.reshape(1,self.env.observation_space.high.size)