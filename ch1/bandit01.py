# -*- coding: utf-8 -*-


import numpy as np
from pandas import DataFrame
import matplotlib
matplotlib.rcParams['font.size'] = 12

class Bandit:
    def __init__(self, arms = 10):
        self.arms = arms
        self.means = np.random.normal(loc = 0.0, scale = 1.0, size = self.arms)
        
    def select(self, arm):
        reward = np.random.normal(loc = self.means[arm], scale = 1.0)
        return reward
        '''
        print(reward)
        print(arm)
        '''
        
def estimate_means(bandit,steps):
    rewards = {}
    for arm in range(bandit.arms):
        rewards[arm] = []
    
    for _ in range(steps):
        arm = np.random.randint(bandit.arms)
        reward = bandit.select(arm)
        rewards[arm].append(reward)
        
    averages = []
    for arm in range(bandit.arms):
        averages.append(sum(rewards[arm])/len(rewards[arm]))
    
    return averages
