from DQN import RNN
from DQN import cartpole
from DQN import Agent_e
import argparse
import os
import random
import time
import logging

import gym
import numpy as np
import keras
import matplotlib.pyplot as plt
import math


STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 2
# RNN
LEARNING_RATE=0.001
EPOCHS = 20
VALIDATION_SPLIT = 0.2
number_of_RNNmodels = 10


def train_RNNmodel(train_states, train_rewards, model):
    # Change reward -100 to label 0 and reward 1 to label 1
    rewards_label = RNN.RNNmodel().change_to_label(train_rewards)
    history = model.fit(train_states, rewards_label, epochs=EPOCHS, verbose=0, batch_size = 1)
    return model, history


def Agent_q_iteration(env, model, target_model, iteration, current_state,
                mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores, rnnModel, number_of_RNNmodels, RNNmodel_1,
                RNNmodel_2, RNNmodel_3, RNNmodel_4, RNNmodel_5, RNNmodel_6, RNNmodel_7, RNNmodel_8, RNNmodel_9,
                                                   RNNmodel_10):
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
    predictions = []
    for j in range(number_of_RNNmodels):
        prediction = rnnModel.predict_RNNmodel(next_state, globals()['RNNmodel_{}'.format(j + 1)])
        predictions.append(prediction)
    if predictions.count(-100) > predictions.count(1):
        reward_pred = -100
    else:
        reward_pred = 1


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

    return action, reward_pred, is_terminal, epsilon, current_state, score, scores



def Agent(t, ratio):
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--num_rand_acts', help="Random actions before learning starts",
                        default=100, type=int)
    parser.add_argument('-s', '--save_after', help="Save after this number of training steps",
                        default=1000, type=int)
    parser.add_argument('-m', '--mem_size', help="Size of the experience replay memory",
                        default=10 ** 4, type=int)
    parser.add_argument('-sn', '--save_name', help="Name of the saved models", default='Agent_e', type=str)
    args = parser.parse_args()

    # Set up logging:
    logging.basicConfig(level=logging.INFO)  # Is this in the right place?
    logger = logging.getLogger(__name__)

    # Other things to modify
    number_training_steps = t
    print_progress_after = 10 ** 2
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

    # TODO: Rename i to iteration, and combined the two loops below. And factor out the random actions loop and the
    #  learning loop into two helper functions.
    # First make some random actions, and initially fill the memories with these:
    test_input = np.zeros((number_random_actions+1,4))
    test_output = np.zeros((number_random_actions+1,1))
    for i in range(number_random_actions + 1):
        iteration = i
        # Random action
        action = env.action_space.sample()
        next_state, reward, is_terminal, _ = env.step(action)
        test_input[i] = next_state
        next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
        if abs(next_state[0]) <= 1.2 and abs(next_state[2]) <= 6*2*math.pi/360:
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
    train_number = 0
    test_number = 0
    training_input = list()
    training_output = list()

    # Create RNN
    rnnModel = RNN.RNNmodel()
    for i in range(number_of_RNNmodels):
        globals()['RNNmodel_{}'.format(i + 1)] = rnnModel.make_RNNmodel()


    plt.ion()
    fig = plt.figure('Agent_r')
    for i in range(number_training_steps):

        iteration = number_random_actions + i

        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        ret = random.random()
        if ret < ratio:
            train_number +=1
            action, reward, is_terminal, epsilon, current_state, score, scores = cartpole.q_iteration(
                env, model, target_model, iteration, current_state,
                mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores)
            training_input.append(next_state)
            training_output.append(reward)
        else:
            test_number +=1
            action, reward_pred, is_terminal, epsilon, current_state, score, scores = \
                Agent_q_iteration(env, model, target_model, iteration, current_state, mem_states, mem_actions,
                                          mem_rewards, mem_terminal, mem_size, score, scores, rnnModel, number_of_RNNmodels, RNNmodel_1,
                                    RNNmodel_2,RNNmodel_3, RNNmodel_4, RNNmodel_5, RNNmodel_6, RNNmodel_7, RNNmodel_8, RNNmodel_9, RNNmodel_10)

        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(scores[::-1][0:100])))
            print('Ratio = {}'.format(train_number/(train_number+test_number)))
            Test_acc = np.zeros((number_of_RNNmodels, 1))
            for j in range(number_of_RNNmodels):
                test_acc, test_loss = rnnModel.test_RNNmodel(test_input, test_output,
                                                             globals()['RNNmodel_{}'.format(j + 1)])
                Test_acc[j] = test_acc
            print('RNN Test mean accuracy:', np.mean(Test_acc))

            for j in range(number_of_RNNmodels):
                globals()['RNNmodel_{}'.format(j + 1)], _ = rnnModel.train_RNNmodel(np.array(training_input),
                                                                                    np.array(training_output),globals()['RNNmodel_{}'.format(j + 1)])
            training_input = list()
            training_output = list()

            plt.clf()
            plt.plot(scores)
            plt.ylabel('scores')
            plt.xlabel('Steps until {}'.format(i + 1))
            plt.pause(0.1)
    plt.ioff()

    #   Save Agent_r
    file_name = os.path.join('Agents', 'Agent_r')
    model.save_weights(file_name)
    print('Agent_r saved')
    return scores

