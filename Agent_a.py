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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from DQN import cartpole
from DQN import RNN
import math

# TODO: Try different loss (mse?) and optimiser (RMSPprop?)
# TODO: Make the choice of environment a command line parameter, so this code is flexible for other environments. Then
#  make things like STATE_SHAPE and ACTION_SIZE general, i.e. use "env.observation_space.shape" and "env.action_space.n"

# State and action sizes *for this particular environment*. These are constants (fixed throughout), so USE_CAPS
STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 3
# RNN
LEARNING_RATE=0.001
EPOCHS = 10
VALIDATION_SPLIT = 0.2
number_of_RNNmodels = 10
number = 7000


def show_max(list):
    index = 0
    max = 0
    for i in range(len(list)):
        count = 0
        for j in range(i+1, len(list)):
            if list[j] == list[i]:
                count +=1
        if count > max:
            max = count
            index = i
    return list[index]



def q_iteration(env, model, target_model, iteration, current_state,
                mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores, rnnModel, RNNmodel_1,
                RNNmodel_2, RNNmodel_3, RNNmodel_4, RNNmodel_5, RNNmodel_6, RNNmodel_7, RNNmodel_8, RNNmodel_9,
                                                   RNNmodel_10, Ask_number,correct_pred):
    """
    Do one iteration of acting then learning
    """
    epsilon = cartpole.get_epsilon_for_iteration(iteration)  # Choose epsilon based on the iteration
    start_state = current_state
    # Choose the action:
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = cartpole.choose_best_action(model, start_state)

    # Play one game iteration: TODO: According to the paper, you should actually play 4 times here
    next_state, _, is_terminal, _ = env.step(action)
    next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
    # Use RNN to predict reward

    IfAsk = False
    predictions = []
    for j in range(number_of_RNNmodels):
        prediction = rnnModel.predict_RNNmodel(next_state, globals()['RNNmodel_{}'.format(j + 1)])
        predictions.append(prediction)
    if predictions.count(1) == number_of_RNNmodels:
        reward_pred = 1
    elif predictions.count(-100) == number_of_RNNmodels:
        reward_pred = -100
    elif predictions.count(2) == number_of_RNNmodels:
        reward_pred = 2
    else:
        IfAsk = True
        reward_pred = None

    # Retrain the RNNmodels
    if IfAsk:
        if is_terminal:
            reward_pred = -100
        elif abs(next_state[0]) <= 1.2 and abs(next_state[2]) <= 6 * 2 * math.pi / 360:
            reward_pred = 2
        else:
            reward_pred = 1
        if show_max(predictions) == reward_pred:
            correct_pred += 1

        for i in range(number_of_RNNmodels):
            globals()['RNNmodel_{}'.format(i + 1)], _ = rnnModel.train_RNNmodel(np.array([next_state]),
                                                                                np.array([reward_pred]),
                                                                                globals()['RNNmodel_{}'.format(i + 1)])
        Ask_number += 1

    score += reward_pred

    # If DONE, reset model, modify reward, record score
    if is_terminal:
        env.reset()
        scores.append(score)  # Record score
        score = 0  # Reset score to zero

    cartpole.add_to_memory(
        iteration+1, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward_pred, is_terminal)

    # Make then fit a batch (gamma=0.99, num_in_batch=32)
    number_in_batch = 32
    cartpole.make_n_fit_batch(model, target_model, 0.99, iteration,
                     mem_size, mem_states, mem_actions, mem_rewards, mem_terminal, number_in_batch)

    current_state = next_state

    return action, reward_pred, is_terminal, epsilon, current_state, score, scores, Ask_number, correct_pred



def Agent(t, ratio):
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
    train_ratio = 0.6
    number_training_steps = round(t * train_ratio)
    number_testing_steps = t - number_training_steps
    print_progress_after = 10**2
    Copy_model_after = 100

    number_random_actions = args.num_rand_acts  # Should be at least 33 (batch_size+1). Is this even needed for Cartpole?
    save_model_after_steps = args.save_after  # Some use 25 here?
    mem_size = args.mem_size  # Some use 2k, or 50k, or 10k?

    logger.info(' num_rand_acts = %s, save_after = %s, mem_size = %s',
                number_random_actions, save_model_after_steps, mem_size)

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

    # First make some random actions, and initially fill the memories with these:
    test_input = np.zeros((number_random_actions + 1, 4))
    test_output = np.zeros((number_random_actions + 1, 1))
    for i in range(number_random_actions + 1):
        iteration = i
        # Random action
        action = env.action_space.sample()
        next_state, reward, is_terminal, _ = env.step(action)
        test_input[i] = next_state
        next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
        if abs(next_state[0]) <= 1.2 and abs(next_state[2]) <= 6 * 2 * math.pi / 360:
            reward = 2

        if is_terminal:
            reward = -100
            env.reset()
            # scores.append(score)  # Record score
            # score = 0  # Reset score to zero
        test_output[i] = reward
        cartpole.add_to_memory(
            iteration, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward, is_terminal)

    # Now do actions using the DQN, and train as we go...
    print('Finished the {} random actions...'.format(number_random_actions))
    tic = 0
    current_state = next_state

    # For recroding the score
    score = 0
    scores = []
    train_input = np.zeros((number_training_steps, 4))
    train_output = np.zeros((number_training_steps, 1))

    plt.ion()
    fig = plt.figure('Agent_a')
    for i in range(number_training_steps):

        iteration = number_random_actions + i

        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        action, reward, is_terminal, epsilon, current_state, score, scores = cartpole.q_iteration(
            env, model, target_model, iteration, current_state,
            mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores)
        train_input[i] = current_state
        train_output[i] = reward

        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(number_random_actions + i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(scores[::-1][0:100])))
            plt.clf()
            plt.plot(scores)
            plt.ylabel('scores')
            plt.xlabel('Steps until {}'.format(number_random_actions + i + 1))
            plt.pause(0.1)
        if (i + 1) % save_model_after_steps == 0:
            toc = time.time()
            print('Time since last save: {}'.format(np.round(toc - tic)), end=" ")
            tic = time.time()
            # Save model:
            file_name = os.path.join('saved_models', 'Run_{}_{}'.format(args.save_name, i + 1))
            model.save(file_name)
            print('; model saved')
    plt.ioff()
    # Create & Train RNN
    for i in range(number_training_steps):
        train_input[i] = mem_states.recall(number_random_actions + 1 + i)

    rnnModel = RNN.RNNmodel()
    for i in range(number_of_RNNmodels):
        rnnmodel = rnnModel.make_RNNmodel()
        globals()['RNNmodel_{}'.format(i + 1)], globals()['history_{}'.format(i + 1)] = rnnModel.train_RNNmodel(
            train_input, train_output, rnnmodel)




    # Now use RNNs:
    env.reset()
    Ask_number = 0
    correct_pred = 0
    for i in range(number_testing_steps):
        iteration = number_training_steps + number_random_actions + i

        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        action, reward_pred, is_terminal, epsilon, current_state, score, scores, Ask_number, correct_pred = q_iteration(
            env, model, target_model, iteration, current_state,mem_states, mem_actions, mem_rewards, mem_terminal,
            mem_size, score, scores, rnnModel, RNNmodel_1, RNNmodel_2, RNNmodel_3, RNNmodel_4, RNNmodel_5, RNNmodel_6,
            RNNmodel_7, RNNmodel_8, RNNmodel_9, RNNmodel_10, Ask_number, correct_pred)


        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(number_random_actions + number_training_steps + i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(scores[::-1][0:100])))
            if Ask_number != 0:
                print('Ask_number = {}, Accuracy = {}'.format(Ask_number, correct_pred/Ask_number))
            Test_acc = np.zeros((number_of_RNNmodels,1))
            for j in range(number_of_RNNmodels):
                test_acc, test_loss = rnnModel.test_RNNmodel(test_input,test_output,globals()['RNNmodel_{}'.format(j + 1)])
                Test_acc[j] = test_acc
            print('RNN Test mean accuracy:', np.mean(Test_acc))
            plt.clf()
            plt.plot(scores)
            plt.ylabel('scores')
            plt.xlabel('Steps until {}'.format(number_random_actions+ number_training_steps + i + 1))
            plt.pause(0.1)
        if (i + 1) % save_model_after_steps == 0:
            toc = time.time()
            print('Time since last save: {}'.format(np.round(toc - tic)), end=" ")
            tic = time.time()
            # Save model:
            file_name = os.path.join('saved_models', 'Run_{}_{}'.format(args.save_name, i + 1))
            model.save(file_name)
            print('; model saved')

    plt.ioff()
    return scores
