from gridworld import DQNTrain, EvoDQNTrain
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import os.path

NUM_EPISODES = 25
AVERAGED_OVER = 10

DQN_TYPE = 0
EVODQN_TYPE = 1


def run_dqn_trials(env_id="MiniGrid-Empty-6x6-v0",
                   dqn_type=DQN_TYPE,
                   num_population=20):
    episode_reward_trials = np.empty((0, NUM_EPISODES))
    base_filename = ('grid_evoreward_pop' + str(num_population)
                     if dqn_type == EVODQN_TYPE else 'grid_dqn')
    base_filename = base_filename + env_id + str(dqn_type)

    log_filename = "logs/" + base_filename

    augment = ''
    trials_completed = 1

    time_start = time.time()

    if dqn_type == EVODQN_TYPE:
        print("DQN with augmented reward----", env_id)
        dqnobject = EvoDQNTrain(
            env=env_id,
            score_averaged_over=AVERAGED_OVER,
            log_file=log_filename)
        [episode_rewards, best_agent_rewards, augment] = dqnobject.train(
            n_episodes=NUM_EPISODES, n_population=num_population)
    else:
        print("DQN with normal reward----", env_id)
        dqnobject = DQNTrain(
            env=env_id,
            score_averaged_over=AVERAGED_OVER,
            log_file=log_filename)
        [episode_rewards] = dqnobject.train(n_episodes=NUM_EPISODES)

    # Save rewards to file as backup
    trial_save_filename = "plotted_values/" + base_filename + 'maxrewards.npy'
    bestagent_save_filename = "plotted_values/" + base_filename + 'agentrewards.npy'

    if os.path.isfile(trial_save_filename):
        old_trials = np.load(trial_save_filename)
        trials_completed = old_trials.shape[0] + 1
        episode_reward_trials = old_trials
    else:
        print('File does not exist yet')

    # append reward history from each trial
    episode_reward_trials = np.append(
        episode_reward_trials, [episode_rewards], axis=0)

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
#run_dqn_trials(env_id = env,dqn_type=EVODQN_TYPE, num_population=n_population)

# plot figure
#mean_rewards = np.average(episode_reward_trials,axis=0)
#std_rewards = np.std(episode_reward_trials,axis=0)
#plt.errorbar(range(1,mean_rewards.size+1),mean_rewards,color='darkorange',ecolor='#FF8C0055',errorevery=5,yerr=std_rewards,label="DQN-evoReward, pop: "+str(n_population))

### Trials DQN
episode_reward_trials = run_dqn_trials(env_id=env)

# plot figure
#mean_rewards = np.average(episode_reward_trials,axis=0)
#std_rewards = np.std(episode_reward_trials,axis=0)
#plt.errorbar(range(1,mean_rewards.size+1),mean_rewards,color='#4682B4FF',ecolor='#4682B455',errorevery=5,yerr=std_rewards,label="DQN")

#plt.xlabel("Episode")
#plt.ylabel("Average Cumulative Reward")
#plt.legend(title="DQN type")
#plt.savefig(base_filename+"result.png")
