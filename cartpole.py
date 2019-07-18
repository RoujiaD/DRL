import argparse
import os
import random
import time
import logging

import gym
import numpy as np
import keras
from keras import layers
from keras.models import Model
import math
import matplotlib.pyplot as plt

# TODO: Try different loss (mse?) and optimiser (RMSPprop?)
# TODO: Make the choice of environment a command line parameter, so this code is flexible for other environments. Then
#  make things like STATE_SHAPE and ACTION_SIZE general, i.e. use "env.observation_space.shape" and "env.action_space.n"

# State and action sizes *for this particular environment*. These are constants (fixed throughout), so USE_CAPS
STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 3


def make_model():
    # With the functional API we need to define the inputs:
    states_input = layers.Input(STATE_SHAPE, name='states')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')  # Masking!

    hidden1 = layers.Dense(32, activation='relu')(states_input)
    hidden2 = layers.Dense(32, activation='relu')(hidden1)
    output = layers.Dense(ACTION_SIZE)(hidden2)  # Activation not specified, so it's linear activation by default?
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])  # Multiply the output by the mask

    model = Model(inputs=[states_input, actions_input], outputs=filtered_output)
    model.compile(loss='logcosh', optimizer=keras.optimizers.Adam(lr=0.001))

    return model


# Choose epsilon based on the iteration
def get_epsilon_for_iteration(iteration):
    # TODO Perhaps: make it so greedy_after can be modified in main (without using global variables!)
    greedy_after = 5000
    # epsilon should be 1 for 0 iterations, 0.1 for greedy_after iterations, and 0.1 from then onwards
    if iteration > greedy_after:
        epsilon = 0.1
    else:
        epsilon = 1 - 0.9*iteration/greedy_after
    return epsilon


# Choose the best action
def choose_best_action(model, state):
    # Need state in correct form/shape
    state_batch = np.zeros((1, ) + STATE_SHAPE)
    state_batch[0,] = state  # TODO: Is this the correct way to select the correct part of state_batch?
    Q_values = model.predict([state_batch, np.ones((1, ACTION_SIZE))])  # Using mask of all ones
    action = np.argmax(Q_values)
    return action


# RingBufSimple: the memory to store the experiences (simplified from the RingBuf in the blog)
# TODO: Make the RingBuf save a whole experience together (not reward, action etc separately). E.g. perhaps something
#  like this: https://github.com/artem-oppermann/Deep-Reinforcement-Learning/blob/master/src/q%20learning/exp_replay.py
class RingBufSimple:
    def __init__(self, size):
        self.data = [None] * size
        self.size = size

    def append(self, element, iteration):
        iteration_mod_size = iteration % self.size  # Turns iteration into an index in the memory [note % means 'mod']
        self.data[iteration_mod_size] = element

    def recall(self, iteration):  # Recalls the memory element in the position corresponding to "iteration"
        iteration_mod_size = iteration % self.size  # Turns iteration into an index in the memory
        return self.data[iteration_mod_size]


# Copying the model
# TODO: Is there no way to copy a model other than saving it to disc?! Yes: I'm using it now...
#def copy_model(model):
#    """Returns a copy of a keras model"""
#    model.save('tmp_model_x')
#    new_model = keras.models.load_model('tmp_model_x')
#    os.remove('tmp_model_x')  # Delete the model once it's been loaded. (Is this working correctly?)
#    return new_model


# Turn the actions into one-hot-encoded actions (required to use the mask)
def into_one_hot(actions):
    one_hot_actions = np.zeros((len(actions), ACTION_SIZE))  # len(actions) returns the batch size
    for i in range(len(actions)):
        for j in range(ACTION_SIZE):
            if j == actions[i]:
                one_hot_actions[i, j] = 1
    return one_hot_actions


def add_to_memory(index, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward, is_terminal):
    # Add to memory
    mem_states.append(next_state, index)  # The state we ended up in after the action
    mem_actions.append(action, index)  # The action we did
    mem_rewards.append(reward, index)  # The reward we received after doing the action
    mem_terminal.append(is_terminal, index)  # Whether or not the new state is terminal


def q_iteration(steps, env, model, target_model, iteration, current_state,
                mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores):
    """
    Do one iteration of acting then learning
    """
    epsilon = get_epsilon_for_iteration(iteration)  # Choose epsilon based on the iteration
    start_state = current_state
    # Choose the action:
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, start_state)
    # Play one game iteration: TODO: According to the paper, you should actually play 4 times here
    next_state, reward, is_terminal, _ = env.step(action)
    steps += 1
    next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
    # Reward shaping
    if abs(next_state[0]) <= 1.2 and abs(next_state[2]) <= 6*2*math.pi/360:
        reward = 2
    if is_terminal:
        reward = -100
    score += reward

    # If DONE, reset model, modify reward, record score

    if is_terminal:
        env.reset()
        steps = 0
        scores.append(score)  # Record score
        score = 0  # Reset score to zero
    elif steps > 200:
        env.reset()
        steps = 0
        scores.append(score)
        score = 0

    add_to_memory(
        iteration+1, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward, is_terminal)

    # Make then fit a batch (gamma=0.99, num_in_batch=32)
    number_in_batch = 32
    make_n_fit_batch(model, target_model, 0.99, iteration,
                     mem_size, mem_states, mem_actions, mem_rewards, mem_terminal, number_in_batch)

    current_state = next_state

    return steps, action, reward, is_terminal, epsilon, current_state, score, scores


def make_n_fit_batch(model, target_model, gamma, iteration,
        mem_size, mem_states, mem_actions, mem_rewards, mem_terminal, number_in_batch):
    """Make a batch then use it to train the model"""

    if iteration < mem_size:
        # Start at 1 because we also need to use the index-1 state
        indices_chosen = random.sample(range(1, iteration), number_in_batch)
    else:
        # Now the memory is full, so we can take any elements from it
        indices_chosen = random.sample(range(0, mem_size), number_in_batch)

    # Initialise the batches
    start_states = np.zeros((number_in_batch, ) + STATE_SHAPE)
    next_states = np.zeros((number_in_batch, ) + STATE_SHAPE)
    actions = np.zeros((number_in_batch))

    rewards = list()  # List rather than array
    is_terminals = list()

    for i in range(len(indices_chosen)):
        index = indices_chosen[i]  # Index corresponds to the iterations
        start_states[i,] = mem_states.recall(index-1)
        next_states[i,] = mem_states.recall(index)
        # NOTE: State given by index was arrived at by taking action given by index & reward received is given by index.
        # In contrast, index-1 labels the previous state, before the action was taken.
        actions[i] = mem_actions.recall(index)
        rewards.append(mem_rewards.recall(index))
        is_terminals.append(mem_terminal.recall(index))

    # We should now have a full batch, which the DNN can train on
    fit_batch_target(model, target_model, gamma, start_states, actions, rewards, next_states)


# Modified slightly from the blog:
def fit_batch_target(model, target_model, gamma, start_states, actions, rewards, next_states):
    """Do one deep Q learning iteration.
    Params:
    - model: The DQN
    - target_model: The target DQN (copied from the DQN every e.g. 10k iterations)
    - gamma: Discount factor (should be 0.99?)
    - start_states: numpy array of starting states
    - actions: numpy array of actions corresponding to the start states (NOT one-hot encoded; normal integer encoding)
    - rewards: List rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    """

    # First, predict the Q values of the next states (using the target model):
    next_Q_values = target_model.predict([next_states, np.ones((len(actions), ACTION_SIZE))])

    # TODO: In e.g. Breakout, the Q value of a terminal state is 0 by definition?? I assume this isn't true for
    #  Cartpole, because we give a reward of -100 for terminal, and -100 < 0! But if Q(terminal)=0 then add this:
    #  "next_Q_values[is_terminals] = 0"

    # The Q values of each start state is the reward + gamma * the max next state Q value:
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)

    one_hot_actions = into_one_hot(actions)  # Turn the actions into one-hot-encoded actions
                                                      # (required to use the mask)
    # Fit the keras model, using the mask:
    model.fit([start_states, one_hot_actions], one_hot_actions * Q_values[:, None],
              epochs=1, batch_size=len(actions), verbose=0)  # Should it be epochs=1??

# def main():
""" Train the DQN to play Cartpole
"""
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--num_rand_acts', help="Random actions before learning starts",
                    default = 100, type=int)
parser.add_argument('-s', '--save_after', help="Save after this number of training steps",
                    default = 10**4, type=int)
parser.add_argument('-m', '--mem_size', help="Size of the experience replay memory",
                    default = 10**4, type=int)
parser.add_argument('-sn', '--save_name', help="Name of the saved models", default=None, type=str)
args = parser.parse_args()

# Set up logging:
logging.basicConfig(level=logging.INFO) # Is this in the right place?
logger = logging.getLogger(__name__)

# Other things to modify
number_training_steps = 10**6
print_progress_after = 10**2
Copy_model_after = 100

number_random_actions = args.num_rand_acts  # Should be at least 33 (batch_size+1). Is this even needed for Cartpole?
save_model_after_steps = args.save_after  # Some use 25 here?
mem_size = args.mem_size  # Some use 2k, or 50k, or 10k?

logger.info(' num_rand_acts = %s, save_after = %s, mem_size = %s',
            number_random_actions, save_model_after_steps, mem_size)

# Make the model
model = make_model()
model.summary()

# Make the memories
mem_states = RingBufSimple(mem_size)
mem_actions = RingBufSimple(mem_size)
mem_rewards = RingBufSimple(mem_size)
mem_terminal = RingBufSimple(mem_size)

print('Setting up Cartpole and pre-filling memory with random actions...')

# Create and reset the Atari env:
env = gym.make('CartPole-v1')
env.reset()
steps = 0

# TODO: Rename i to iteration, and combined the two loops below. And factor out the random actions loop and the
#  learning loop into two helper functions.
# First make some random actions, and initially fill the memories with these:
for i in range(number_random_actions+1):
    iteration = i
    # Random action
    action = env.action_space.sample()
    next_state, reward, is_terminal, _ = env.step(action)
    steps += 1
    next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
    if abs(next_state[0]) <= 1.2 and abs(next_state[2]) <= 6 * 2 * math.pi / 360:
        reward = 2

    if is_terminal:
        reward = -100
        env.reset()
    elif steps >200:
        env.reset()
        steps = 0

    add_to_memory(
        iteration, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward, is_terminal)

# Now do actions using the DQN, and train as we go...
print('Finished the {} random actions...'.format(number_random_actions))
tic = 0
current_state = next_state

# For recroding the score
score = 0
scores = []
plt.ion()
fig = plt.figure('Agent_f')
for i in range(number_training_steps):

    iteration = number_random_actions + i

    # Copy model periodically and fit to this: this makes the learning more stable
    if i % Copy_model_after == 0:
        target_model = keras.models.clone_model(model)
        target_model.set_weights(model.get_weights())

    steps, action, reward, is_terminal, epsilon, current_state, score, scores = q_iteration(
        steps, env, model, target_model, iteration, current_state,
        mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores)

    # Print progress, time, and SAVE the model
    if (i + 1) % print_progress_after == 0:
        print('Training steps done: {}, Epsilon: {}'.format(i + 1, epsilon))
        print('Mean score = {}'.format(np.mean(scores)))
        print('Average scores for last 100 trials = {}'.format(np.mean(scores[::-1][0:100])))
        plt.clf()
        plt.plot(scores)
        plt.ylabel('scores')
        plt.xlabel('Steps until {}'.format(i + 1))
        plt.pause(0.1)

    if (i + 1) % save_model_after_steps == 0:
        toc = time.time()
        print('Time since last save: {}'.format(np.round(toc - tic)), end=" ")
        tic = time.time()
        # Save model:
        file_name = os.path.join('saved_models', 'Run_{}_{}'.format(args.save_name, i + 1))
        model.save(file_name)
        print('; model saved')

    if np.mean(scores[::-1][0:100]) > 120:
        print('T-max = {}'.format(i+1))
        break
plt.ioff()
#   Save Agent_f
file_name = os.path.join('Agents', 'Agent_f')
model.save_weights(file_name)
print('Agent_f saved')
    # return i+1

# if __name__ == '__main__':
#     main()