import numpy as np
import gym
from DQN import cartpole
from DQN import Agent_e
from DQN import Agent_r

from DQN import Agent_a

import os
import matplotlib.pyplot as plt

iterations = 1000
T_max = 15000
ratio = 0.8
# T_max=np.zeros((5,1))
# for i in range(5):
#     t = cartpole.main()
#     T_max[i] = t
# T_max = np.mean(T_max)

# scores_Agent_a = Agent_a.Agent(T_max, ratio)
scores_Agent_r = Agent_r.Agent(T_max, ratio)
scores_Agent_e = Agent_e.Agent(T_max, ratio)
Agent_e = cartpole.make_model()
Agent_e.load_weights(os.path.join('Agents', 'Agent_e'))
# Agent_r = cartpole.make_model()
# Agent_r.load_weights(os.path.join('Agents', 'Agent_r'))
Agent_a = cartpole.make_model()
Agent_a.load_weights(os.path.join('Agents', 'Agent_a'))

env = gym.make('CartPole-v1')
state = env.reset()
score = 0
scores = []
plt.ion()
fig = plt.figure(1)
for j in range(iterations):
    action = cartpole.choose_best_action(Agent_e,state)
    next_state, reward, is_terminal, _ = env.step(action)
    next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
    score += reward
    state = next_state

    # If DONE, reset model, modify reward, record score
    if is_terminal:
        reward = -100
        state = env.reset()
        scores.append(score)  # Record score
        score = 0  # Reset score to zero
    plt.clf()
    plt.plot(scores)
    plt.ylabel('scores')
    plt.xlabel('Steps until {}'.format(j + 1))
    plt.pause(0.1)

plt.ioff()
np.meana(scores)