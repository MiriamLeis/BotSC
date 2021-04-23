import numpy as np
import os

from abstract_agent import AbstractAgent

class QAgent(AbstractAgent):
    def __init__(self, agent, total_episodes):
        super().__init__()
        self.agent = agent

        info = self.agent.get_info()

        self.actions = info[1]
        self.lr = info[2]
        self.gamma = info[3]
        self.total_episodes = total_episodes

        self.states_list = set()
        self.q_table = np.zeros((0, len(self.actions))) # create Q table

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
    
    '''
        Q-Learning algorithm
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
        Return action with maxium reward
    '''
    def get_max_action(self, env):
        idx = list(self.states_list).index(self.current_state)
        q_values = self.q_table[idx]
        self.action = int(np.argmax(q_values))
        self.agent.get_action(env=env, action=self.action)

    '''
        Choose action for current state.
        This action could be random or the one with maxium reward, depending on epsilon value.
    '''
    def choose_action(self, env):
        self.__check_state_exist(state=self.current_state)
            
        if np.random.rand() > self.epsilon:
            self.action = np.random.choice(self.actions)
            self.agent.get_action(env=env, action=self.action)
        else:
            self.get_max_action(env=env)

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