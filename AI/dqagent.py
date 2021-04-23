import keras
import os
import itertools
import random

import numpy as np

from keras import Model
from keras.layers import Dense, Input
from collections import deque

from abstract_agent import AbstractAgent

class DQAgent(AbstractAgent):
    def __init__(self, agent, total_episodes):
        super().__init__()
        self.agent = agent
        
        info = self.agent.get_info()

        self.num_actions = info[1]
        self.discount = info[3]
        self.rep_mem_size = info[4]
        self.min_rep_mem_size = info[5]
        self.minibatch_size = info[6]
        self.update_time = info[7]
        self.max_cases = info[8]
        self.cases_to_delete = info[9]
        self.total_episodes = total_episodes

        self.min_rep_mem_total = self.min_rep_mem_size

        # main model
        # model that we are not fitting every step
        # gets trained every step
        self.model = self.__create_model(num_states=info[2], hidden_nodes=info[10], num_hidden_layers=info[11])

        # target model
        # this is what we .predict against every step
        self.target_model = self.__create_model(num_states=info[2], hidden_nodes=info[10], num_hidden_layers=info[11])
        self.target_model.set_weights(self.model.get_weights()) # do it again after a while

    '''
        Return basic information
    '''
    def get_info(self):
        return self.agent.get_info()

    '''
        Prepare agent for next episode
    '''
    def prepare(self, env, ep):
        self.__set_epsilon(episode=ep)
        self.action = self.agent.prepare(env)
        self.current_state = self.agent.get_state(env)

    '''
        Update basic values
    '''
    def update(self, env, deltaTime):
        self.new_state = self.agent.get_state(env)
        self.reward = self.agent.get_reward(env, self.action)
        self.agent.update(env, deltaTime)
        
    '''
        Update basic values
    '''
    def late_update(self, env, deltaTime):
        self.current_state = self.new_state

    '''
        Do step of the environment
        Return PySC2 environment obs
    '''
    def step(self, env, environment):
        return self.agent.step(env=env,environment=environment)

    def train(self):
        self.__update_replay_memory(self, transition=(self.current_state, self.action, self.reward, self.new_state))

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
        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.discount * max_future_q

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)



        history = self.model.fit(np.array(X), np.array(y), batch_size=self.min_rep_mem_size, verbose=0, 
            shuffle=False)

        #updating to determinate if we want to update target_model yet
        self.target_update_counter += 1

        # copy weights from the main network to target network
        if self.target_update_counter > self.update_time:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    '''
        Return action with maxium reward
    '''
    def get_max_action(self, env):
        cases = self.__get_qs(self.current_state)
        self.action = np.argmax(cases)
        self.agent.get_action(env=env, action=self.action)

    '''
        Choose action for current state.
        This action could be random or the one with maxium reward, depending on epsilon value.
    '''
    def choose_action(self, env):
        if np.random.rand() > self.epsilon:
            self.action = np.random.choice(self.actions)
            self.agent.get_action(env=env, action=self.action)
        else:
            self.get_max_action(env=env)

    '''
        Save models to specify file
    '''              
    def save(self, filepath):
        keras.models.save_model(self.model, os.getcwd() + filepath + '.h5')

    '''
        Load models from specify file
    '''        
    def load(self, filepath):
        self.model = keras.models.load_model(os.getcwd() + filepath + '.h5')
        self.target_model = keras.models.load_model(os.getcwd() + filepath + '.h5')
    
    '''
        (Private method)
        Create target model and network model with specify characteristics
    '''
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

    '''
        (Private method)
        Update replay memory with our observation space
    '''
    def __update_replay_memory(self, transition):
        # transition -> our observation space
        self.replay_memory.append(transition)

    '''
        (Private method)
        Predict current state
    '''
    def __get_qs(self, state):
        # return last layer of neural network
        stateArray = np.array(self.current_state)
        return self.model.predict(stateArray.reshape(-1, *stateArray.shape))[0]

    '''
        (Private method)
        Update epsilon value
    '''
    def __set_epsilon(self, episode):
        self.epsilon = (episode / (self.total_episodes - (self.total_episodes / 2)))