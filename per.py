# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:51:55 2019

@author: Andrei DAMIAN
"""

import numpy as np
from collections import deque, namedtuple
import random

from time import time

_VER_ = '0.9.1'

class SumTree(object):
  """
  This SumTree code is modified version of Morvan Zhou: 
  https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
  """
  data_pointer = 0
  stored_data = 0
  
  """
  Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
  """
  def __init__(self, capacity):
      self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
      
      # Generate the tree with all nodes values = 0
      # To understand this calculation (2 * capacity - 1) look at the schema above
      # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
      # Parent nodes = capacity - 1
      # Leaf nodes = capacity
      self.tree = np.zeros(2 * capacity - 1)
      
      """ tree:
          0
         / \
        0   0
       / \ / \
      0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
      """
      
      # Contains the experiences (so the size of data is capacity)
      self.data = np.zeros(capacity, dtype=object)
  
  
  """
  Here we add our priority score in the sumtree leaf and add the experience in data
  """
  def add(self, priority, data):
      # Look at what index we want to put the experience
      tree_index = self.data_pointer + self.capacity - 1
      
      """ tree:
          0
         / \
        0   0
       / \ / \
tree_index  0 0  0  We fill the leaves from left to right
      """
      
      # Update data frame
      self.data[self.data_pointer] = data
      
      # Update the leaf
      self.update (tree_index, priority)
      
      # Add 1 to data_pointer
      self.data_pointer += 1
      
      if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
          self.data_pointer = 0
          
      self.stored_data += 1
          
  
  """
  Update the leaf priority score and propagate the change through tree
  """
  def update(self, tree_index, priority):
      # Change = new priority score - former priority score
      change = priority - self.tree[tree_index]
      self.tree[tree_index] = priority
      
      # then propagate the change through tree
      while tree_index != 0:    # this method is faster than the recursive loop in the reference code
          
          """
          Here we want to access the line above
          THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
          
              0
             / \
            1   2
           / \ / \
          3  4 5  [6] 
          
          If we are in leaf at index 6, we updated the priority score
          We need then to update index 2 node
          So tree_index = (tree_index - 1) // 2
          tree_index = (6-1)//2
          tree_index = 2 (because // round the result)
          """
          tree_index = (tree_index - 1) // 2
          self.tree[tree_index] += change
  
  
  """
  Here we get the leaf_index, priority value of that leaf and experience associated with that index
  """
  def get_leaf(self, v):
      """
      Tree structure and array storage:
      Tree index:
           0         -> storing priority sum
          / \
        1     2
       / \   / \
      3   4 5   6    -> storing priority for experiences
      Array type for storing:
      [0,1,2,3,4,5,6]
      """
      parent_index = 0
      
      while True: # the while loop is faster than the method in the reference code
          left_child_index = 2 * parent_index + 1
          right_child_index = left_child_index + 1
          
          # If we reach bottom, end the search
          if left_child_index >= len(self.tree):
              leaf_index = parent_index
              break
          
          else: # downward search, always search for a higher priority node
              
              if v <= self.tree[left_child_index]:
                  parent_index = left_child_index
                  
              else:
                  v -= self.tree[left_child_index]
                  parent_index = right_child_index
          
      data_index = leaf_index - self.capacity + 1

      return leaf_index, self.tree[leaf_index], self.data[data_index]

  def get_leafs(self, values):
    indices = []
    priorities = []
    datas = []    
    for v in values:
      idx, prior, data = self.get_leaf(v)
      if 'int' in str(type(data)):
        print("\nWARNING: No obs leaf: data:{}  v:{}  idx:{}  priority:{}".format(
            data, v, idx, prior))
        continue
      indices.append(idx)
      priorities.append(prior)
      datas.append(data)
    return np.array(indices), np.array(priorities), datas
    
  
  @property
  def total_priority(self):
      return self.tree[0] # Returns the root node
    
    
class GenericReplayBuffer(object):
  def __init__(self,  capacity, engine='torch', device=None, continuous=False):
    self.capacity = capacity
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.engine = engine
    self.continuous = continuous
    if engine == 'torch' and device is None:
      raise ValueError("Must provide device if engine is torch")
    self.device = device
    self.episode = -1
    self.debug_cpu_copy = []
    self.cpu_start = 0
    self.cpu_end = 0
    self.__version__ = _VER_
    print("Init GRB v.{}".format(self.__version__))
    return
  

  def start_cpu_copy(self):
    self.cpu_start = time()
  
  def end_cpu_copy(self):
    self.cpu_end = time()
    self.debug_cpu_copy.append(self.cpu_end - self.cpu_start)
    return
  
  def get_cpu_copy_time(self):
    return np.sum(self.debug_cpu_copy)  
  
  def _prepare_experience_buffer(self, experience_buffer):
    np_states = np.vstack([e.state for e in experience_buffer if e is not None])
    np_actions = np.vstack([e.action for e in experience_buffer if e is not None])
    np_rewards = np.vstack([e.reward for e in experience_buffer if e is not None])
    np_next_states = np.vstack([e.next_state for e in experience_buffer if e is not None])
    np_dones = np.vstack([e.done for e in experience_buffer if e is not None]).astype(np.uint8)
    if self.engine == 'torch':
      self.start_cpu_copy()
      import torch as th
      states = th.from_numpy(np_states).float().to(self.device)
      if self.continuous:
        actions = th.from_numpy(np_actions).float().to(self.device)
      else:
        actions = th.from_numpy(np_actions).long().to(self.device)
      rewards = th.from_numpy(np_rewards).float().to(self.device)
      next_states = th.from_numpy(np_next_states).float().to(self.device)
      dones = th.from_numpy(np_dones).float().to(self.device)
      self.end_cpu_copy()
    else:
      states = np_states
      actions = np_actions
      rewards = np_rewards
      next_states = np_next_states
      dones = np_dones
    return (states, actions, rewards, next_states, dones)
    
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.store(e)
    return

  def store(self, experience):
    raise ValueError("Called abstract method!")
    return
    
    
class PERMemory(GenericReplayBuffer):  # stored as ( s, a, r, s_ ) in SumTree
  """
  This SumTree code is modified version and the original code is from:
  https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
  """

  def __init__(self, **kwargs):
    # Making the tree 
    """
    Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
    And also a data array
    We don't use deque because it means that at each timestep our experiences change index by one.
    We prefer to use a simple array and to overwrite when the memory is full.
    """
    super().__init__(**kwargs)    
    self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    self.PER_b_increment_per_sampling = 0.001
    
    self.absolute_error_upper = 1.  # clipped abs error
    self.tree = SumTree(self.capacity)
    return

    
  """
  Store a new experience in our tree
  Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
  """
  def store(self, experience):
    # Find the max priority
    max_priority = np.max(self.tree.tree[-self.tree.capacity:])
    
    # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
    # So we use a minimum priority
    if max_priority == 0:
        max_priority = self.absolute_error_upper
    
    self.tree.add(max_priority, experience)   # set the max p for new p
    return
    
      
  """
  - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
  - Then a value is uniformly sampled from each range
  - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
  - Then, we calculate IS weights for each minibatch element
  """
  def _sample_original(self, n):
    # Create a sample array that will contains the minibatch
    memory_buff = []
    
    b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
    
    # Calculate the priority segment
    # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
    priority_segment = self.tree.total_priority / n       # priority segment

    # Here we increasing the PER_b each time we sample a new minibatch
    self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
    
    # Calculating the max_weight that would be the weight 
    # of the smallest priority (if unlikely sampled)
    p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
    max_weight = (p_min * n) ** (-self.PER_b)
    
    for i in range(n):
      """
      A value is uniformly sample from each range
      """
      a, b = priority_segment * i, priority_segment * (i + 1)
      value = np.random.uniform(a, b)
      
      """
      Experience that correspond to each value is retrieved
      """
      index, priority, data = self.tree.get_leaf(value)
      
      #P(j)
      sampling_proba = priority / self.tree.total_priority
      
      #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
      b_ISWeights[i, 0] = np.power(n * sampling_proba, -self.PER_b)/ max_weight
                             
      b_idx[i]= index
      
      experience = data #[data]
      
      memory_buff.append(experience)
  
    (states, actions, rewards, next_states, dones) = self._prepare_experience_buffer(memory_buff)
    
    if self.engine == 'torch':
      import torch as th
      b_ISWeights = th.from_numpy(b_ISWeights).float().to(self.device)
    
    return (states, actions, rewards, next_states, dones), b_idx, b_ISWeights


  def sample(self, n_samples):
    # Create a sample array that will contains the minibatch
        
    # Calculate the priority segment
    # Here, as explained in the paper, we divide the Range[0, ptotal] into n_samples ranges
    priority_segment = self.tree.total_priority / n_samples       # priority segment

    # Here we increasing the PER_b each time we sample a new minibatch
    self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
    
    values = []
    for i in range(n_samples):
      """
      A value is uniformly sample from each range
      """
      _a, _b = priority_segment * i, priority_segment * (i + 1)
      value = np.random.uniform(_a, _b)
      values.append(value)
    
    indices, priorities, datas = self.tree.get_leafs(values)
    
    # no need to pow(p, alpha) as we already did that ad update
    sampling_probas = priorities / self.tree.total_priority
    
    N = self.tree.stored_data
    
    weights = np.power(N * sampling_probas, -self.PER_b)
    max_weight = weights.max()
    
    np_IS_weights = weights / max_weight
  
    (states, actions, rewards, next_states, dones) = self._prepare_experience_buffer(datas)
    
    np_IS_weights = np_IS_weights.reshape((-1,1))
    
    if self.engine == 'torch':
      import torch as th
      out_IS_weights = th.from_numpy(np_IS_weights).float().to(self.device)
    else:
      out_IS_weights = np_IS_weights
    
    return (states, actions, rewards, next_states, dones), indices, out_IS_weights

  
  
  """
  Update the priorities on the tree
  """
  def batch_update(self, tree_idx, abs_errors):
    """
    assumes abs_errors is > 0
    """
    abs_errors += self.PER_e  # add eps to avoid zero
    clipped_errors = np.minimum(abs_errors, self.absolute_error_upper) # upper bound 1.
    ps = np.power(clipped_errors, self.PER_a)
    
    for ti, p in zip(tree_idx, ps):
      self.tree.update(ti, p)    
      
      
  def __len__(self,):
    return min(self.tree.stored_data, self.tree.capacity)


class NaivePrioritizedBuffer(GenericReplayBuffer):
  def __init__(self, prob_alpha=0.6, beta_start=0.4, **kwargs):
    super().__init__(**kwargs)
    self.prob_alpha = prob_alpha
    self.buffer     = []
    self.pos        = 0
    self.beta       = beta_start
    self.beta_increment = 0.001
    self.priorities = np.zeros((self.capacity,), dtype=np.float32)
    
  def store(self, experience):    
    max_prio = self.priorities.max() if self.buffer else 1.0    

    if len(self.buffer) < self.capacity:
        self.buffer.append(experience)
    else:
        self.buffer[self.pos] = experience
    
    self.priorities[self.pos] = max_prio
    self.pos = (self.pos + 1) % self.capacity
    return
  
  def sample(self, batch_size):
    if len(self.buffer) == self.capacity:
        prios = self.priorities
    else:
        prios = self.priorities[:self.pos]
    
    probs  = prios ** self.prob_alpha
    probs /= probs.sum()
    
    indices = np.random.choice(len(self.buffer), batch_size, p=probs)
    samples = [self.buffer[idx] for idx in indices]
    
    self.beta = np.min([1., self.beta + self.beta_increment]) 
    
    total    = len(self.buffer)
    weights  = (total * probs[indices]) ** (-self.beta)
    weights /= weights.max()
    weights  = np.array(weights, dtype=np.float32).reshape((-1,1))
    
    (states, actions, rewards, next_states, dones) = self._prepare_experience_buffer(samples)

    if self.engine == 'torch':
      import torch as th
      out_IS_weights = th.from_numpy(weights).float().to(self.device)
    else:
      out_IS_weights = weights
    
    return (states, actions, rewards, next_states, dones) , indices, out_IS_weights
  
  def batch_update(self, batch_indices, batch_priorities):
    for idx, prio in zip(batch_indices, batch_priorities):
        self.priorities[idx] = np.minimum(prio + 1e-5, 1.)

  def __len__(self):
    return len(self.buffer)
      
      
class SimpleReplayBuffer(GenericReplayBuffer):
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, seed=1234, **kwargs):
    """Initialize a ReplayBuffer object.

    Params
    ======
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed
    """
    super().__init__(**kwargs)
    self.memory = deque(maxlen=self.capacity)  
    self.seed = random.seed(seed)
    return
  
  def store(self, experience):
    self.memory.append(experience)
  

  def sample(self, n):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=n)
    states, actions, rewards, next_states, dones = self._prepare_experience_buffer(experiences)
    return (states, actions, rewards, next_states, dones)

  def __len__(self):
      """Return the current size of internal memory."""
      return len(self.memory)
      
if __name__ == '__main__':
  ###
  ### now lets test with some env and random agent
  ###   
  pass