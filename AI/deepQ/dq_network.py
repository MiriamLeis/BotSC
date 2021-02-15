import keras
import tensorflow as tf
import numpy as np

from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import Adam
from collections import deque

MODEL_NAME = '1_64_4'

class DQNAgent:
    def __init__(self, num_actions, num_states, discount=0.99, rep_mem_size=50_000, min_rep_mem_size=50, update_time=5):
        #parameters
        self.num_actions = num_actions
        self.num_states = num_states
        self.discount = discount
        self.rep_mem_size = rep_mem_size
        self.min_rep_mem_size = min_rep_mem_size
        self.update_time = update_time

        # main model
        # model that we are not fitting every step
        # gets trained every step
        self.model = self.create_model()

        # target model
        # this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights()) # do it again after a while

        # deque -> array or list 
        self.replay_memory = deque(maxlen=self.rep_mem_size)

        # track internally when we are ready to update target_model
        self.target_update_counter = 0

    def create_model(self):
        # layers
        inputs = Input(shape=(self.num_states,))
        x = Dense(25, activation='relu')(inputs)
        outputs = Dense(self.num_actions, activation='softmax')(x)

        # creation
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
    
        model.summary()
        return model

    def update_replay_memory(self, transition):
        # transition -> our observation space
        self.replay_memory.append(transition)

    def get_qs(self, state):
        # return last layer of neural network
        stateArray = np.array(state)
        return self.model.predict(stateArray.reshape(-1, *stateArray.shape))[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.min_rep_mem_size:
            return

        # / 255 cuz we want to normalize in this case... but this is just for images.
        # so this is cuz we want values from 0 to 1
        current_states = np.array([transition[0] for transition in self.replay_memory])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in self.replay_memory])
        future_qs_list = self.target_model.predict(new_current_states)

        X = [] # feature sets   /  images
        y = [] # labels         /  actions

        # current_states get from index 0 so current_state has to be in 0 position
        # same with new_current_state
        ## ALGORITHM
        for index, (current_state, action, reward, new_current_state, done) in enumerate(self.replay_memory):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        # we do fit only if terminal_state, otherwise we fit None
        self.model.fit(np.array(X), np.array(y), batch_size=self.min_rep_mem_size, verbose=2, 
            shuffle=False)

        self.replay_memory.clear()

        #updating to determinate if we want to update target_model yet
        
        self.target_update_counter += 1

        if self.target_update_counter > self.update_time:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0