from multiprocessing.managers import BaseManager
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

class EvoRewardManager(BaseManager):
    pass

EvoRewardManager.register('evoReward', evoReward)

class Test:
    def __init__(self):

        self.manager = EvoRewardManager()

    def train(self):
        self.manager.start()
        shared = self.manager.evoReward(DNA_bound=REWARD_BOUND,
                                        cross_rate=CROSS_RATE,
                                        mutation_rate=MUTATION_RATE,
                                        pop_size=n_population,
                                        obs_space_size=10,
                                        action_size=2,
                                        dna_type=1)

        file_write_q = mp.Queue()

        pool = mp.Pool(mp.cpu_count()+2)

        watcher = pool.apply_async(self.listener, (file_write_q))

        for n in range(n_population):
            pool.apply_async(self.process, (shared, file_write_q, n))

        file_write_q.put('kill')
        pool.close()

    def process(self, evo_reward_obj, file_write_q, agent_id):
        file_write_q.put(" {} starting outer".format(agent_id))
        for n in range(10):
            evo_reward_obj.notify_episode_start()
            file_write_q.put(" {} starting inner".format(agent_id))
            time.sleep(np.random.randint(2,10))
            file_write_q.put(" {} starting inner".format(agent_id))

        evo_reward_obj.set_agent_fitness(np.random.randint(0, 50), agent_id)

        while not evo_reward_obj.are_rewards_ready():
            time.sleep(1)

        file_write_q.put(" {} exiting outer".format(agent_id))

    def listener(self, q):
        '''listens for messages on the q, writes to file. '''

        f = open("log.txt", 'wb')
        while 1:
            m = q.get()
            if m == 'kill':
                break
            f.write(str(m) + '\n')
            f.flush()
        f.close()

obj = Test()
obj.train()


