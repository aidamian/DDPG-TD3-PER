# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:34:27 2019

@author: Andrei
"""
import numpy as np
import gym

class SimpleMultiEnv:
  
  def __init__(self, env_name, nr_workers):
    envs = []
    for i in range(nr_workers):
      envs.append(gym.make(env_name))
    self.envs = envs
    self.n_envs = len(envs)
    self.n_workers = self.n_envs
    self.obs_size = envs[0].observation_space.shape[0]
    self.act_size = envs[0].action_space.shape[0]
    return
    
  def step(self, actions):
    assert len(actions) == self.n_envs
    next_states, rewards, dones = [], [], []
    for i in range(self.n_envs):
      next_state, reward, done, _ = self.envs[i].step(actions[i])
      next_states.append(next_state)
      rewards.append(reward)
      dones.append(done)
    return np.array(next_states), np.array(rewards), np.array(dones)
  
  def reset(self):
    states = []
    for i in range(self.n_envs):
      states.append(self.envs[i].reset())
    return np.array(states)