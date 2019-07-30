import gym

import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
from collections import deque
import random

import torch as th

from agent import Agent
from simple_multi_worker_env import SimpleMultiEnv

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=130)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)     

def sample_action(num_agents, action_size):
  actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
  actions = np.clip(actions, -1, 1)   
  return actions



def reset_seed(seed=123):
  """
  radom seeds reset for reproducible results
  """
  print("Resetting all seeds...", flush=True)
  random.seed(seed)
  np.random.seed(seed)
  th.manual_seed(seed)
  return
    

def training_loop(multi_env, agent, n_episodes=1200, max_t=1000,
                  noise_scaling_reduction=False,
                  policy_noise_reduction=False, explor_noise_reduction=False,
                  stop_policy_noise=0, stop_explor_noise=0,
                  train_every_steps=10,
                  DEBUG=1,
                 ):    
  print("Starting training for {} episodes...".format(n_episodes))
  print("  explor_noise_reduction:   {}".format(explor_noise_reduction))
  print("  policy_noise_reduction:   {}".format(policy_noise_reduction))
  print("  stop_policy_noise:        {}".format(stop_policy_noise))
  print("  stop_explor_noise:        {}".format(stop_explor_noise))
  print("  noise_scaling_reduction:  {}".format(noise_scaling_reduction))
  solved_episode = 0
  _LOST = 0
  _WINS = 0
  scores_deque = deque(maxlen=100)
  steps_deque = deque(maxlen=100)
  max_ep_scores = deque(maxlen=100)
  scores_avg = []
  scores = []
  ep_times = []
  t_start_training = time.time()
  for i_episode in range(1, n_episodes+1):
    t_start = time.time()
    states = multi_env.reset()
    np_score = np.zeros(multi_env.n_envs)        
    for t in range(max_t):
      if agent.is_warming_up():
          actions = sample_action(multi_env.n_envs, multi_env.act_size)
      else:
          actions = agent.act(states, add_noise=True)
      next_states, rewards, dones = multi_env.step(actions)
      agent.step(states, actions, rewards, next_states, dones, train_every_steps=train_every_steps)
      states = next_states
      np_score += rewards
      if np.any(dones):
        # break bool if any worker finished the episode
        first_done = np.argmax(dones)
        done_reward = rewards[first_done]
        done_score = np_score[first_done]
        break  
      
    if done_reward == -100:
      _end = 'CRSHED'
      _LOST += 1
    else:
      _end = 'LANDED'
      _WINS += 1
    
    if train_every_steps==0:
      agent.train(nr_iters=t//10)
    
    if noise_scaling_reduction:
      agent.reduce_noise_scaling()
      
    episode_max = np_score.max()  
    max_ep_scores.append(episode_max)
    score = np_score.mean()
    scores_deque.append(score)
    max_score = np.max(scores_deque)
    scores.append(score)
    scores_avg.append(np.mean(scores_deque))
    steps_deque.append(t)
    t_end = time.time()
    ep_time = t_end - t_start
    ep_times.append(ep_time)
    _cl1 = "{:>8.1e}".format(np.mean(agent.critic_1_losses)) if agent.critic_1_losses else "NA"
    _cl2 = "{:>8.1e}".format(np.mean(agent.critic_2_losses)) if agent.critic_2_losses else "NA"
    _al = "{:>8.1e}".format(np.mean(agent.actor_losses)) if agent.actor_losses else "NA"
    print('\rEpisode {:>4}  LS/Me/Ma100/Avg: {:>6.1f}/{:>6.1f}/{:>6.1f}/{:>6.1f}  Stp/W/Scr: {:>4}/{}/{:.1f}  [μcL1/μcL2: {}/{} μaL: {}]  t:{:>4.1f}s'.format(
        i_episode, score, np.max(max_ep_scores), max_score, np.mean(scores_deque), t, _end, done_score, _cl1,_cl2, _al, ep_time), end="", flush=True)
    if (np.mean(scores_deque) >= 200.0) and (solved_episode == 0):
        print("\nEnvironment solved at episode {}!".format(i_episode))
        agent.save('ep_{}_solved'.format(i_episode))
        solved_episode = i_episode
        break
    if i_episode % 100 == 0:
        mean_ep = np.mean(ep_times)
        elapsed = i_episode * mean_ep
        total = (n_episodes + 1) * mean_ep
        left_time_hrs = (total - elapsed) / 3600     
        total_elapsed = time.time() - t_start_training
        non_mandatory_cpu_copies = agent.get_cpu_copy_time()
        print('\rEpisode {:>4}  LS/Me100/Ma100/Avg: {:>6.1f}/{:>6.1f}/{:>6.1f}/{:>6.1f}  AvStp:{:>4.0f} W/L: {:>3}/{:>3}  [c1/c2/a {}/{}/{}]  tlft:{:>4.1f} h eT:{:>4.2f} h nmCPU: {:.2f}s'  .format(
            i_episode, score, episode_max, max_score, np.mean(scores_deque), np.mean(steps_deque), _WINS, _LOST,
            _cl1,_cl2, _al, left_time_hrs, total_elapsed / 3600, non_mandatory_cpu_copies))
        _WINS = 0
        _LOST = 0
        if DEBUG >= 1:
            print("  Loaded steps: {:>10} (Replay memory: {})".format(agent.step_counter, len(agent.memory)))
            print("  Critic/Actor updates:  {:>10} / {:>10}".format(agent.train_iters, agent.actor_updates))
        if DEBUG >= 2:
            agent.debug_weights()
        if explor_noise_reduction:
            agent.reduce_explore_noise(0.8)
        if policy_noise_reduction:
            agent.reduce_policy_noise(0.8)
    if stop_policy_noise>0 and i_episode >= stop_policy_noise:
        agent.clear_policy_noise()
    if stop_explor_noise>0 and i_episode >= stop_explor_noise:
        agent.clear_explore_noise()
            
  #agent.save('ep_{}'.format(i_episode))
  return scores, scores_avg, solved_episode



if __name__ == '__main__':
  
  num_envs = 8
  
  dev = th.device("cuda:0" if th.cuda.is_available() else "cpu")
  DEBUG = 0

  reset_seed()
  
  
  multi_env = SimpleMultiEnv(env_name='LunarLanderContinuous-v2', 
                             nr_workers=num_envs)

  
  iterations = [
                'TD3_PER_t_hub',
                'TD3_PER_t',
                'DDPG_PER_t_hub',
                'DDPG_PER_t',
                'DDPG_s_nsr', 
                'DDPG_s', 
                'TD3_s_nsr', 
                'TD3_s', 
                'DDPG_PER_t',
                'TD3_PER_s_hub',
                'TD3_PER_s',
                ]

  results = {
      "AGENT"    : [],
      "EP2SOL"   : [],
      "BEST_AVG" : [],
      }
  all_iters = []
  
  for ii, iteration_name in enumerate(iterations):
    print("\n\nStarting grid search iteration {}/{}:'{}'".format(
        ii+1, len(iterations), iteration_name))
    
    train_every_steps = 1
          
  
    use_td3 = 'TD3' in iteration_name
    if 'PER' in iteration_name:
      if 'PER_t' in iteration_name:        
        per = 'tree_per'
      else:
        per = 'naive_per'
    else:
      per = None
      
    if 'nsr' in iteration_name:
      start_noise_scaling = 2.
      noise_scaling_reduction = True
    else:
      start_noise_scaling = 1.
      noise_scaling_reduction = False
    
    if 'hub' in iteration_name:
      use_huber = True
    else:
      use_huber = False
      
      
              
    agent = Agent(a_size=multi_env.act_size, 
                  s_size=multi_env.obs_size, 
                  dev=dev, TD3=use_td3, PER=per,
                  n_env_agents=multi_env.n_workers,
                  start_noise_scaling=start_noise_scaling,
                  name=iteration_name, 
                  huber_loss=use_huber,
                  show_models=False,
                 )
    _res = training_loop(multi_env=multi_env, agent=agent,
                         noise_scaling_reduction=noise_scaling_reduction,
                         train_every_steps=train_every_steps,
                         DEBUG=DEBUG,
                        )
    scores, scores_avg, i_solved = _res
    results['AGENT'].append(iteration_name)
    results['BEST_AVG'].append(np.max(scores_avg))
    results['EP2SOL'].append(i_solved)
    all_iters.append((iteration_name, scores_avg))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores,"-b", label='score')
    plt.plot(np.arange(1, len(scores)+1), scores_avg,"-r", label='average')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.title(iteration_name)
    plt.axhline(y=200, linestyle='--', color='green')
    plt.savefig(iteration_name+'.png')
    plt.show()

    df_res = pd.DataFrame(results).sort_values(['EP2SOL','BEST_AVG'])
    print(df_res)
    df_res.to_csv('results.csv')

    
  clrs_opt = ['b','g','r','c','m','y']
  styles_opt = ['-','--',':']
  styles = [x+y for x in styles_opt for y in clrs_opt]
  plt.figure(figsize=(15,10))
  for ii, done_iter in enumerate(all_iters):
      iter_name = done_iter[0]
      avg_scores = done_iter[1]
      plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, styles[ii], label=iter_name)
  plt.legend()
  plt.title("Results comparison")
  plt.ylabel('Avg. Score')
  plt.xlabel('Episode #')
  plt.axhline(y=30, linestyle='--', color='black')
  plt.savefig('comparison.png')
  plt.show()
        

