import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# State and action sizes *for this particular environment*. These are constants (fixed throughout), so USE_CAPS
STATE_SHAPE = (4,) # This is the shape after pre-processing: "state = np.array([state])"
ACTION_SIZE = 3
# RNN
LEARNING_RATE=0.001
EPOCHS = 20
VALIDATION_SPLIT = 0.2
number_of_RNNmodels = 10
number = 1000




class RNNmodel:

    def __init__(self):
        self.state_shape = STATE_SHAPE
        self.action_size = ACTION_SIZE


    def make_RNNmodel(self):

        model = Sequential()
        model.add(Dense(32, input_shape=self.state_shape, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation=tf.nn.softmax))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE),metrics=['accuracy'])
        return model

    def change_to_label(self, rewards):
        rewards_label = np.ones((len(rewards), 1))
        for i in range(len(rewards)):
            if rewards[i] == -100:
                rewards_label[i] = 0
            if rewards[i] == 2:
                rewards_label[i] = 2
        return rewards_label

    def change_to_single(self, output):
        y = np.ones((len(output), 1))
        for i in range(len(output)):
            a = np.argmax(output)
            if a == 0:
                y[i] = -100
            elif a == 2:
                y[i] = 2
        return y

    def predict_RNNmodel(self, test_states, RNNmodel):
        predict_label = RNNmodel.predict(np.array([test_states]))
        prediction = self.change_to_single(predict_label)[0][0]
        return prediction

    def train_RNNmodel(self, train_states, train_rewards, model):
        # Change reward -100 to label 0 and reward 1 to label 1
        rewards_label = self.change_to_label(train_rewards)
        history = model.fit(train_states, rewards_label, epochs=EPOCHS, verbose=0)
        return model, history

    def test_RNNmodel(self, test_states, test_output, model):
        output_label = self.change_to_label(test_output)
        test_loss, test_acc = model.evaluate(test_states, output_label,verbose=0)
        return test_acc, test_loss
