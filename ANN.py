import numpy as np
import gym
from DQN import cartpole
from DQN import Agent_e
from DQN import Agent_r
from DQN import Agent_f
from DQN import Agent_a

import os
import matplotlib.pyplot as plt

iterations = 1000

T_max=np.zeros((5,1))
for i in range(5):
    t = cartpole.main()
    T_max[i] = t
T_max = round(np.mean(T_max))

scores_Agent_f = Agent_f.Agent(T_max)
# Use Agent_a to determine ratio
scores_Agent_a, ratio = Agent_a.Agent(T_max)
scores_Agent_r = Agent_r.Agent(T_max, ratio)
scores_Agent_e = Agent_e.Agent(T_max, ratio)
Agent_e = cartpole.make_model()
Agent_e.load_weights(os.path.join('Agents', 'Agent_e'))
Agent_r = cartpole.make_model()
Agent_r.load_weights(os.path.join('Agents', 'Agent_r'))
Agent_a = cartpole.make_model()
Agent_a.load_weights(os.path.join('Agents', 'Agent_a'))
Agent_f = cartpole.make_model()
Agent_f.load_weights(os.path.join('Agents', 'Agent_f'))

# Plot scores
plt.figure('Scores Summary')
plt.xlabel('Number of trials')
plt.ylabel('Scores')
plt.title('Scores Summary, Ration = {}'.format(ratio))
plt.plot(scores_Agent_f, label = 'Agent_f, mean score (last 100 trials) = {}'.format(np.mean(scores_Agent_f[::-1][0:100])))
plt.plot(scores_Agent_a, label = 'Agent_a, mean score (last 100 trials) = {}'.format(np.mean(scores_Agent_a[::-1][0:100])))
plt.plot(scores_Agent_e, label = 'Agent_e, mean score (last 100 trials) = {}'.format(np.mean(scores_Agent_e[::-1][0:100])))
plt.plot(scores_Agent_r, label = 'Agent_r, mean score (last 100 trials) = {}'.format(np.mean(scores_Agent_r[::-1][0:100])))
plt.legend()

# Test
env = gym.make('CartPole-v1')
agents = [Agent_f,Agent_a,Agent_e,Agent_r]
for i in range(len(agents)):
    state = env.reset()
    steps = 0
    score = 0
    globals()['scores_{}'.format(i + 1)] = []

    for j in range(iterations):
        action = cartpole.choose_best_action(agents[i],state)
        next_state, reward, is_terminal, _ = env.step(action)
        steps += 1
        next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
        score += reward
        state = next_state

        # If DONE, reset model, modify reward, record score
        if is_terminal:
            reward = -100
            env.reset()
            globals()['scores_{}'.format(i + 1)].append(score)  # Record score
            score = 0  # Reset score to zero
        elif steps > 200:
            globals()['scores_{}'.format(i + 1)].append(score)
            score = 0
            steps = 0
            env.reset()

fig = plt.figure('Testing Agents')
plt.plot(scores_1, label='Agent_f, mean score = {}'.format(np.mean(scores_1)))
plt.plot(scores_2, label='Agent_a, mean score = {}'.format(np.mean(scores_2)))
plt.plot(scores_3, label='Agent_e, mean score = {}'.format(np.mean(scores_3)))
plt.plot(scores_4, label='Agent_r, mean score = {}'.format(np.mean(scores_4)))
plt.legend()
plt.ylabel('scores')
plt.xlabel('Number of trials')
plt.title('Testing Agents')


