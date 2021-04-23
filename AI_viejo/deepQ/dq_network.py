import keras
import tensorflow as tf
import numpy as np
import random
import itertools
import os

from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import Adam
from collections import deque
from matplotlib import pyplot as plt

class DQNAgent:
    def __init__(self, num_actions, num_states, episodes,discount=0.99, rep_mem_size=50_000, min_rep_mem_size=50, learn_every = 50, update_time=100, minibatch_size=25, max_cases=300, cases_to_delete = 0, hidden_nodes=25, num_hidden_layers = 1, load = False):
        #parameters
        self.num_actions = num_actions
        self.total_episodes = episodes
        self.discount = discount
        self.rep_mem_size = rep_mem_size
        self.min_rep_mem_size = learn_every
        self.min_rep_mem_total = min_rep_mem_size
        self.minibatch_size = minibatch_size
        self.update_time = update_time
        self.max_cases = max_cases
        self.cases_to_delete = cases_to_delete

        self.accuracy = np.empty(0)
        self.loss = np.empty(0)
        self.steps = np.empty(0)

        if not load:

            # main model
            # model that we are not fitting every step
            # gets trained every step
            self.model = self.__create_model(num_states, hidden_nodes, num_hidden_layers)

            # target model
            # this is what we .predict against every step
            self.target_model = self.__create_model(num_states, hidden_nodes, num_hidden_layers)
            self.target_model.set_weights(self.model.get_weights()) # do it again after a while

        # deque -> array or list 
        self.replay_memory = deque(maxlen=self.rep_mem_size)

        # track internally when we are ready to update target_model
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        # transition -> our observation space
        self.replay_memory.append(transition)

    def learn(self, step, ep = 0):

        if len(self.replay_memory) < self.min_rep_mem_total:
            return
        if len(self.replay_memory) > self.max_cases:
            # delete leftovers 
            self.replay_memory = deque(itertools.islice(self.replay_memory, self.cases_to_delete, None))

        self.min_rep_mem_total += self.min_rep_mem_size

        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = [] # feature sets   /  images
        y = [] # labels         /  actions

        # current_states get from index 0 so current_state has to be in 0 position
        # same with new_current_state
        ## ALGORITHM
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)



        history = self.model.fit(np.array(X), np.array(y), batch_size=self.min_rep_mem_size, verbose=0, 
            shuffle=False)

        self.accuracy = np.append(self.accuracy, history.history['accuracy'])
        self.loss = np.append(self.loss, history.history['loss'])
        self.steps = np.append(self.steps, step + (ep*1900))

        #updating to determinate if we want to update target_model yet
        self.target_update_counter += 1

        # copy weights from the main network to target network
        if self.target_update_counter > self.update_time:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
  
    def get_max_action(self, state):
            cases = self.__get_qs(state)
            print("Estado : ", state)
            print("Acciones : ", cases)
            return np.argmax(cases)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            # get maxium reward action
            return self.get_max_action(state)
        else:
            # get random action
            return np.random.randint(0, self.num_actions)
        
    def set_epsilon(self, episode):
        self.epsilon = 1 - (episode / (self.total_episodes - (self.total_episodes / 2)))

    def save(self, filepath):
        keras.models.save_model(self.model, os.getcwd() + '\\deepQ\\models\\' + filepath + '.h5')

    def loadModel(self, filepath):
        self.model = keras.models.load_model(filepath)
        self.target_model = keras.models.load_model(filepath)

    def saveGraphics(self):
        plt.figure()
        plt.plot(self.steps, self.accuracy, color='royalblue')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('steps')
        plt.savefig('accuracy.svg')

        plt.figure()
        plt.plot(self.steps, self.loss, color='royalblue')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('steps')
        plt.savefig('loss.svg')

    def __create_model(self, num_states, hidden_nodes = 25, num_hidden_layers = 1):
        
        # layers
        inputs = Input(shape=(num_states,))
        x = Dense(hidden_nodes, activation='relu')(inputs)
        for i in range(1, num_hidden_layers):
            x = Dense(hidden_nodes + (20 * i), activation='relu')(x)
        outputs = Dense(self.num_actions)(x)

        # creation
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
    
        model.summary()
        return model
    
    def __get_qs(self, state):
        # return last layer of neural network
        stateArray = np.array(state)
        return self.model.predict(stateArray.reshape(-1, *stateArray.shape))[0]