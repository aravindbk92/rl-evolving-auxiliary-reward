#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:36:51 2018

@author: a4balakr
"""

import numpy as np

TYPE_INDEX = 0 # if dna consists of explicit rewards that mapped to action x state pairs
TYPE_WEIGHTS = 1 # if dna consists of weights mapped to actions. reward is linear combination of weights and state

class evoReward():
    def __init__(self, DNA_bound, cross_rate, mutation_rate, pop_size, obs_space_size, action_size, dna_type=TYPE_INDEX):
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.fitness = np.array([])
        self.obs_space_size = obs_space_size
        self.action_size = action_size
        self.dna_type = dna_type
        
        self.DNA_size = obs_space_size * action_size
        self.pop = np.random.randint(*DNA_bound, size=(pop_size, self.DNA_size)).astype(np.int8)

    def set_fitness(self, fitness):
        ptp = np.ptp(fitness)
        if ptp != 0:
            normalized_fitness = (fitness - np.min(fitness))/ptp
            self.fitness = normalized_fitness
        else:
            self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def select(self):
        fitness = self.get_fitness() + 1e-4     # add a small amount to avoid all zero fitness
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(*self.DNA_bound)  # choose a random ASCII index
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop
        
    def get_reward(self, dna_index, action, state):
        # if dna consists of explicit rewards that mapped to action x state pairs
        if self.dna_type == TYPE_INDEX:
            reward_index = action*self.obs_space_size + state
            return self.pop[dna_index,reward_index]
        
        # if dna consists of weights mapped to actions
        # reward is linear combination of weights and state
        if self.dna_type == TYPE_WEIGHTS:
            weight_matrix = self.pop[dna_index].reshape(self.action_size,self.obs_space_size)
            weights = weight_matrix[action]
            return np.sum(weights * state)
    
    def get_DNA(self, n):
        return self.pop[n].reshape((self.action_size,self.obs_space_size))