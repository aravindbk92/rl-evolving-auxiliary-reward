import numpy as np
import matplotlib.pyplot as plt
import gym
import math

env = gym.make('CartPole-v0')
print (env.observation_space.shape[0])