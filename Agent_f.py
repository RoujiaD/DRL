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
from DQN import cartpole

# TODO: Try different loss (mse?) and optimiser (RMSPprop?)
# TODO: Make the choice of environment a command line parameter, so this code is flexible for other environments. Then
#  make things like STATE_SHAPE and ACTION_SIZE general, i.e. use "env.observation_space.shape" and "env.action_space.n"

# State and action sizes *for this particular environment*. These are constants (fixed throughout), so USE_CAPS
STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 2



def Agent(t):
    """ Train the DQN to play Cartpole
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--num_rand_acts', help="Random actions before learning starts",
                        default = 100, type=int)
    parser.add_argument('-m', '--mem_size', help="Size of the experience replay memory",
                        default = 10**4, type=int)
    args = parser.parse_args()

    # Set up logging:
    logging.basicConfig(level=logging.INFO) # Is this in the right place?
    logger = logging.getLogger(__name__)

    # Other things to modify
    number_training_steps = t
    print_progress_after = 10**2
    Copy_model_after = 100

    number_random_actions = args.num_rand_acts  # Should be at least 33 (batch_size+1). Is this even needed for Cartpole?
    mem_size = args.mem_size  # Some use 2k, or 50k, or 10k?

    logger.info(' num_rand_acts = %s, mem_size = %s',
                number_random_actions, mem_size)

    # Make the model
    model = cartpole.make_model()
    model.summary()

    # Make the memories
    mem_states = cartpole.RingBufSimple(mem_size)
    mem_actions = cartpole.RingBufSimple(mem_size)
    mem_rewards = cartpole.RingBufSimple(mem_size)
    mem_terminal = cartpole.RingBufSimple(mem_size)

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

        cartpole.add_to_memory(
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

        steps, action, reward, is_terminal, epsilon, current_state, score, scores = cartpole.q_iteration(
            steps, env, model, target_model, iteration, current_state,
            mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores)

        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(scores[::-1][0:100])))
            plt.clf()
            plt.plot(scores)
            plt.title('Agent_f')
            plt.ylabel('scores')
            plt.xlabel('Number of Trials (Steps until {})'.format(i + 1))
            plt.pause(0.1)

    plt.ioff()

    # Save Agent_a
    file_name = os.path.join('Agents', 'Agent_f')
    model.save(file_name)
    print('Agent_f saved')

    return scores