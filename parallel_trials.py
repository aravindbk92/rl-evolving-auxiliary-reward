from evoDQN import DQNTrain, EvoDQNTrain
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import os.path
import gym
import gym_minigrid

NUM_EPISODES = 25
AVERAGED_OVER = 10
DQN_TYPE = 0
EVODQN_TYPE = 1

class GridWorldEnvironmentWrapper:
    def __init__(self):
        self.env_name = "MiniGrid-Empty-16x16-v0"
        self.env = gym.make(self.env_name)
        self.obs_space_size = self.env.observation_space.spaces['image'].shape[0] ** 2
        self.action_space_size = self.env.action_space.n

    def preprocess(self, state):
        return state['image'][:, :, 0].reshape(1, self.obs_space_size)

class AtariEnvironmentWrapper:
    def __init__(self):
        self.env_name = "CartPole-v0"
        self.env = gym.make(self.env_name)
        self.obs_space_size = self.env.observation_space.shape[0]
        self.action_space_size = self.env.action_space.n

    def preprocess(self, state):
        return state.reshape(1,self.obs_space_size)

# Runs a trial in parallel. Use run_grid_trials.bash
def run_dqn_trials(env_id="MiniGrid-Empty-6x6-v0",
                   dqn_type=DQN_TYPE,
                   num_population=20):

    # Log and stat file names
    base_filename = base_filename + env_id + str(dqn_type)
    base_filename = ('grid_evoreward_pop' + str(num_population)
                     if dqn_type == EVODQN_TYPE else 'grid_dqn')
    trial_save_filename = "plotted_values/" + base_filename + 'maxrewards.npy'
    bestagent_save_filename = "plotted_values/" + base_filename + 'agentrewards.npy'
    log_filename = "logs/" + base_filename

    # initial vars
    env_wrapper = GridWorldEnvironmentWrapper()
    episode_reward_trials = np.empty((0, NUM_EPISODES))
    augment = ''
    trials_completed = 1

    # Start train
    time_start = time.time()
    if dqn_type == EVODQN_TYPE:
        print("DQN with augmented reward----", env_id)
        dqnobject = EvoDQNTrain(
            env_wrapper=env_wrapper,
            score_averaged_over=AVERAGED_OVER,
            log_file=log_filename)
        [episode_rewards, best_agent_rewards, augment] = dqnobject.train(
            n_episodes=NUM_EPISODES, n_population=num_population)
    else:
        print("DQN with normal reward----", env_id)
        dqnobject = DQNTrain(
            env_wrapper=env_wrapper,
            score_averaged_over=AVERAGED_OVER,
            log_file=log_filename)
        [episode_rewards] = dqnobject.train(n_episodes=NUM_EPISODES)

    # Save rewards to file as backup
    if os.path.isfile(trial_save_filename):
        old_trials = np.load(trial_save_filename)
        trials_completed = old_trials.shape[0] + 1
        episode_reward_trials = old_trials
    else:
        print('File does not exist yet')

    # append reward history from each trial
    episode_reward_trials = np.append(
        episode_reward_trials, [episode_rewards], axis=0)

    # Save stats as files
    np.save(trial_save_filename, episode_reward_trials)
    if dqn_type == EVODQN_TYPE:
        np.save(bestagent_save_filename, best_agent_rewards)

    # find time taken for each trial
    time_end = time.time()
    time_taken = (time_end - time_start) / 60

    # find mean and std
    mean_rewards = np.average(episode_reward_trials, axis=0)
    std_rewards = np.std(episode_reward_trials, axis=0)

    # Save stats to log
    with open(log_filename, 'a+') as f:
        print(file=f)

        print(
            ">> Rewards over last " + str(trials_completed) + " trial(s)",
            file=f)
        print("Last:", file=f)
        print(episode_reward_trials, file=f)
        print("Mean:", file=f)
        print(mean_rewards, file=f)
        print("Std:", file=f)
        print(std_rewards, file=f)

        print("Best augment:", file=f)
        print(augment, file=f)

        print(file=f)
        print(">> Time:", file=f)
        print("Total time taken: ", str(time_taken), file=f)
        print('>>>>Finished ' + env_id + ' in ' + str(time_taken) + " minutes")
        print()

    return episode_reward_trials

parser = argparse.ArgumentParser("simple_example")
parser.add_argument(
    "--env",
    help="String with env id",
    type=str,
    default="MiniGrid-Empty-6x6-v0")
parser.add_argument(
    "--n_pop", help="Number of population of agents", type=int, default=10)
args = parser.parse_args()
env = args.env
n_population = args.n_pop

# Trials DQN with evoReward
run_dqn_trials(env_id = env,dqn_type=EVODQN_TYPE, num_population=n_population)

# Trials DQN
episode_reward_trials = run_dqn_trials(env_id=env)

# Use load_and_generate_figure.py to generate graphs from saved episode_reward_trials