from DQN import RNN
from DQN import cartpole
from DQN import Agent_a
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
import matplotlib.pyplot as plt
import math


STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 2
# RNN
LEARNING_RATE=0.001
EPOCHS = 20
VALIDATION_SPLIT = 0.2
number_of_RNNmodels = 10
number = 1000


# Copying the model
# TODO: Is there no way to copy a model other than saving it to disc?! Yes: I'm using it now...
#def copy_model(model):
#    """Returns a copy of a keras model"""
#    model.save('tmp_model_x')
#    new_model = keras.models.load_model('tmp_model_x')
#    os.remove('tmp_model_x')  # Delete the model once it's been loaded. (Is this working correctly?)
#    return new_model


def Agent_q_iteration(steps, env, model, target_model, iteration, current_state,
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
    steps += 1
    next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
    # Use RNN to predict reward
    predictions = []
    for j in range(number_of_RNNmodels):
        prediction = rnnModel.predict_RNNmodel(next_state, globals()['RNNmodel_{}'.format(j + 1)])
        predictions.append(prediction)
    reward_pred = Agent_a.show_max(predictions)

    score += reward_pred

    # If DONE, reset model, modify reward, record score
    if is_terminal:
        env.reset()
        scores.append(score)  # Record score
        score = 0  # Reset score to zero
    elif steps > 200:
        env.reset()
        steps = 0
        scores.append(score)
        score = 0

    cartpole.add_to_memory(
        iteration+1, mem_states, mem_actions, mem_rewards, mem_terminal, next_state, action, reward_pred, is_terminal)

    # Make then fit a batch (gamma=0.99, num_in_batch=32)
    number_in_batch = 32
    cartpole.make_n_fit_batch(model, target_model, 0.99, iteration,
                     mem_size, mem_states, mem_actions, mem_rewards, mem_terminal, number_in_batch)

    current_state = next_state

    return steps, action, reward_pred, is_terminal, epsilon, current_state, score, scores



def Agent(t,ratio):
    t_train = round(t * ratio)
    t_test = round(t * (1 - ratio))
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--num_rand_acts', help="Random actions before learning starts",
                        default=100, type=int)
    parser.add_argument('-m', '--mem_size', help="Size of the experience replay memory",
                        default=10 ** 4, type=int)
    args = parser.parse_args()

    # Set up logging:
    logging.basicConfig(level=logging.INFO)  # Is this in the right place?
    logger = logging.getLogger(__name__)

    # Other things to modify
    number_training_steps = t_train
    number_testing_steps = t_test
    print_progress_after = 10 ** 2
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
    test_input = np.zeros((number_random_actions+1,4))
    test_output = np.zeros((number_random_actions+1,1))
    for i in range(number_random_actions + 1):
        iteration = i
        # Random action
        action = env.action_space.sample()
        next_state, reward, is_terminal, _ = env.step(action)
        steps += 1
        test_input[i] = next_state
        next_state = np.array([next_state])[0, :]  # Process state so that it's a numpy array, shape (4,)
        if abs(next_state[0]) <= 1.2 and abs(next_state[2]) <= 6 * 2 * math.pi / 360:
            reward = 2

        if is_terminal:
            reward = -100
            env.reset()
            # scores.append(score)  # Record score
            # score = 0  # Reset score to zero
        elif steps > 200:
            env.reset()
            steps = 0

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
    training_input = list()
    training_output = list()
    # Create RNN
    rnnModel = RNN.RNNmodel()
    for i in range(number_of_RNNmodels):
        globals()['RNNmodel_{}'.format(i + 1)] = rnnModel.make_RNNmodel()

    plt.ion()
    fig = plt.figure('Agent_e')
    for i in range(number_training_steps):

        iteration = number_random_actions + i

        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        steps, action, reward, is_terminal, epsilon, current_state, score, scores = cartpole.q_iteration(
            steps, env, model, target_model, iteration, current_state,
            mem_states, mem_actions, mem_rewards, mem_terminal, mem_size, score, scores)
        training_input.append(current_state)
        training_output.append(reward)

        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(scores[::-1][0:100])))
            for j in range(number_of_RNNmodels):
                globals()['RNNmodel_{}'.format(j + 1)], _ = rnnModel.train_RNNmodel(np.array(training_input),
                                                                                    np.array(training_output),globals()['RNNmodel_{}'.format(j + 1)])
            training_input = list()
            training_output = list()
            plt.clf()
            plt.plot(scores,label = 'train')
            plt.ylabel('scores')
            plt.xlabel('Steps until {}'.format(i + 1))
            plt.pause(0.1)

    Test_acc = np.zeros((number_of_RNNmodels, 1))
    for j in range(number_of_RNNmodels):
        test_acc, test_loss = rnnModel.test_RNNmodel(test_input, test_output,
                                                     globals()['RNNmodel_{}'.format(j + 1)])
        Test_acc[j] = test_acc
    print('RNN Test mean accuracy:', np.mean(Test_acc))

    # Use RNN for the rest of steps


    for i in range(number_testing_steps):
        iteration = number_training_steps + number_random_actions + i
        # Copy model periodically and fit to this: this makes the learning more stable
        if i % Copy_model_after == 0:
            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())

        steps, action, reward_pred, is_terminal, epsilon, current_state, score, scores = \
            Agent_q_iteration(steps, env, model, target_model, iteration, current_state,mem_states, mem_actions, mem_rewards,
                              mem_terminal, mem_size, score, scores, rnnModel, number_of_RNNmodels, RNNmodel_1,RNNmodel_2, RNNmodel_3,
                              RNNmodel_4, RNNmodel_5, RNNmodel_6, RNNmodel_7, RNNmodel_8, RNNmodel_9,RNNmodel_10)

        # Print progress, time, and SAVE the model
        if (i + 1) % print_progress_after == 0:
            print('Training steps done: {}, Epsilon: {}'.format(number_training_steps + i + 1, epsilon))
            print('Mean score = {}'.format(np.mean(scores)))
            print('Average scores for last 100 trials = {}'.format(np.mean(scores[::-1][0:100])))
            plt.clf()
            plt.plot(scores, label='test', color='tomato')
            plt.plot(scores[0:number_training_steps + number_random_actions + 1], label='train')
            plt.legend()
            plt.title('Agent_e')
            plt.ylabel('scores')
            plt.xlabel('Number of trials (Steps until {})'.format(number_training_steps + i + 1))
            plt.pause(0.1)

    plt.ioff()
#   Save Agent_e
    file_name = os.path.join('Agents', 'Agent_e')
    model.save(file_name)
    print('Agent_e saved')
    return scores
