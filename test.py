import numpy as np
import matplotlib.pyplot as plt

not_shared = np.load("not_shared.npy")
shared = np.load("shared.npy")

plt.plot(not_shared, label="Not shared")
plt.plot(shared, label="Shared")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend(title="Sharing Actor-Critic First Layer")
plt.savefig("ddpg.png")
