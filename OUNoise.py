import numpy as np
import torch as th


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

  def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
    assert type(action_dimension) == tuple
    self.action_dimension = action_dimension
    self.scale = scale
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.state = np.ones(self.action_dimension) * self.mu
    self.reset()

  def reset(self):
    self.state = np.ones(self.action_dimension) * self.mu

  def noise(self, return_torch=False):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.action_dimension)
    self.state = x + dx
    if return_torch:
      return th.tensor(self.state * self.scale).float()
    else:
      return self.state * self.scale
        
        
class OUNoiseOriginal:
  """Ornstein-Uhlenbeck process."""

  def __init__(self, size, seed=None, mu=0., theta=0.15, sigma=0.2, dt=1e-2):
    """Initialize parameters and noise process."""
    assert type(size) == tuple
    self.size = size
    self.mu = mu * np.ones(size)
    self.theta = theta
    self.sigma = sigma
    self.dt = dt
    if seed:
      self.seed = np.random.seed(seed)
    self.reset()

  def reset(self):
    """Reset the internal state (= noise) to mean (mu)."""
    self.state = self.mu.copy()

  def sample(self):
    """Update internal state and return it as a noise sample."""
    x = self.state
    dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(*self.size)
    self.state = x + dx
    return self.state        