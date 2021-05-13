import numpy as np
import os

from agents.abstract_agent import AbstractAgent

class QAgent(AbstractAgent):
    def __init__(self, info):
        self.learning = info['learn']
        self.total_episodes = info['episodes']

        self.actions = info['actions']
        self.lr = info['learning_rate']
        self.gamma = info['gamma']

        self.states_list = set()
        self.q_table = np.zeros((0, len(self.actions))) # create Q table

    '''
        Prepare agent for next episode
    '''
    def prepare(self, env, episode):
        self.__set_epsilon(episode=episode)
        self.action = 0
        self.current_state = env.get_state()

    '''
        Update basic values
    '''
    def update(self, env, deltaTime):
        self.new_state = env.get_state()
        self.reward = env.get_reward(self.action)
        env.update(deltaTime)

        # train
        if self.learning:
            self.train()

        # late update
        self.current_state = self.new_state

        # get action
        if self.learning:
            self._choose_action(env=env)
        else:
            self._get_max_action(env=env)
    
    '''
        Execute Q-Learning algorithm
    '''
    def train(self):
        self.__check_state_exist(self.new_state)
        self.__check_state_exist(self.current_state)

        state_idx = list(self.states_list).index(self.current_state)
        next_state_idx = list(self.states_list).index(self.new_state)

        q_predict = self.q_table[state_idx, self.action]

        q_target = self.reward + (self.gamma * self.q_table[next_state_idx].max())

        self.q_table[state_idx, self.action] = ((1 - self.lr) * q_predict) + (self.lr * (q_target))
        
    '''
        Save Q-Table and States to specify file
    '''
    def save(self, filepath):
        #q-table
        np.save(filepath + '_qtable', self.q_table)
        #states
        temp = np.array(list(self.states_list))
        np.save(filepath + '_states', temp)

    '''
        Load Q-Table and States from specify file
    '''
    def load(self, filepath):
        #q-table
        self.q_table = np.load(os.getcwd() + filepath + '_qtable.npy')
        #states
        tmp_array = np.load(os.getcwd() + filepath + '_states.npy')
        for x in tmp_array:
            self.states_list.add(x)

    '''
        (Protected method)
        Return action with maxium reward
    '''
    def _get_max_action(self, env):
        idx = list(self.states_list).index(self.current_state)
        q_values = self.q_table[idx]
        self.action = int(np.argmax(q_values))
        env.get_action(action=self.action)

    '''
        (Protected method)
        Choose action for current state.
        This action could be random or the one with maxium reward, depending on epsilon value.
    '''
    def _choose_action(self, env):
        self.__check_state_exist(state=self.current_state)
            
        if np.random.rand() > self.epsilon:
            self.action = np.random.choice(self.actions)
            env.get_action(action=self.action)
        else:
            self._get_max_action(env=env)

    '''
        (Private method)
        Check if current state exist
    '''
    def __check_state_exist(self, state):
        if state not in self.states_list:
            self.q_table = np.vstack([self.q_table, np.zeros((1, len(self.actions)))])
            self.states_list.add(state)

    '''
        (Private method)
        Update epsilon value
    '''
    def __set_epsilon(self, episode):
        self.epsilon = episode / (self.total_episodes - (self.total_episodes / 3))