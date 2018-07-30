import numpy as np
import matplotlib.pyplot as plt
import gym
import math

env = gym.make('CartPole-v0')

CON = 195
def fun(x):
    #y =  1-np.exp(2*((x-CON)/CON))
    y = (1-(x/CON))**2
    return y
    
fn = np.vectorize(fun)
x = np.array(range(195))

plt.plot(fn(x))