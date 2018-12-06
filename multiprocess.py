from multiprocessing.managers import SyncManager
from multiprocessing import Process
import multiprocessing as mp
from evoReward import evoReward
import time
import numpy as np

CROSS_RATE = 0.4  # mating probability (DNA crossover)
MUTATION_RATE = 0.01  # mutation probability
REWARD_BOUND = [-5, 5]  # Bounds for the augmented reward
EPSILON = 0.3
PRINT_INTERVAL = 1
LOGGING_MEAN_SIZE = 5
REWARD_MULTIPLIER = 100
SOLVE_SCORE = 0.90
n_population = 8

class EvoRewardManager(SyncManager):
    pass

EvoRewardManager.register('evoReward', evoReward)

def multiprocess():
    manager = EvoRewardManager()
    manager.start()

    shared = manager.evoReward(DNA_bound=REWARD_BOUND,
                                cross_rate=CROSS_RATE,
                                mutation_rate=MUTATION_RATE,
                                pop_size=n_population,
                                obs_space_size=10,
                                action_size=2)

    processes = []
    for n in range(n_population):
        p = Process(target=process, args=[shared, n])
        processes.append(p)
        p.start()

    for one_process in processes:
        one_process.join()

def process(evo_reward_obj, agent_id):
    n_episodes = 2
    for n in range(n_episodes):
        evo_reward_obj.notify_episode_start()
        time.sleep(np.random.randint(0.1,3))

        evo_reward_obj.set_agent_fitness(np.random.randint(0, 50), agent_id)

        while not evo_reward_obj.are_rewards_ready():
            time.sleep(1)

if __name__ == '__main__':
    multiprocess()