import numpy as np
import math
import random
import os

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

from defeatzealots_2enemies import Agent as internal_agent

'''
    Add these parameters mandatory
'''

# environment values

MAP_NAME = 'DefeatZealotswithBlink'
FILE_NAME = 'zealots2vs2Model'
EPISODES = 1000


'''
    Agent class must have this methods:

        class Agent:
            def preprare(obs)
            def update(obs, deltaTime)
            def get_num_actions()
            def get_num_states()
            def get_state(obs)
            def get_action(obs)
            def check_action_available(self, obs, action, func)
            def get_reward(obs, action)
            def get_end(obs)
            def check_done(obs, last_step)
'''

class Agent:
    '''
        Initialize the agent
    '''
    def __init__(self, load=False):
        self.select_1 = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])
        self.select_2 = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])

        self.agent_1 = internal_agent()
        self.agent_2 = internal_agent()
        
        if load:
            self.agent_1.loadModel(self, os.getcwd() + '\\deepQ\\models\\' + FILE_NAME + '_1.h5')
            self.agent_2.loadModel(self, os.getcwd() + '\\deepQ\\models\\' + FILE_NAME + '_2.h5')

    '''
        Prepare basic parameters. This is called before start the episode.
    '''
    def prepare(self, obs, episode):
        _, state_1 = self.agent_1.prepare(obs, episode)
        _, state_2 = self.agent_2.prepare(obs, episode)

        # first action is trash
        return actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL]), (state_1, state_2)

    '''
        Do step of the environment
    '''
    def step(self, env, func):
        if not self.end_1:
            obs = env.step(actions=[self.select_1])
            self.end_1 = self.agent_1.get_end(obs[0])

            if not self.end_1:
                obs = self.agent_1.step(env, actions=[func[0]])

        self.end_2 = self.agent_2.get_end(obs[0])
        if not self.end_2:
            obs = env.step(actions=[self.select_2])
            self.end_2 = self.agent_2.get_end(obs[0])

            if not self.end_2:
                obs = self.agent_2.step(env, actions=[func[1]])

        return obs, self.get_end(obs)

    '''
        Update basic values
    '''
    def update(self, obs, delta):
        self.agent_1.update(obs, delta)
        self.agent_2.update(obs, delta)
    
    '''
        Train agent
    '''
    def train(self, step, current_state, action, reward, new_state, done):
        self.agent_1.train(step, current_state[0], action[0], reward[0], new_state[0], done[0])
        self.agent_2.train(step, current_state[1], action[1], reward[1], new_state[1], done[1])

    '''
        Return agent number of actions
    '''
    def get_num_actions(self):
        return (self.agent_1.get_num_actions(), self.agent_2.get_num_actions())
    
    '''
        Return agent number of states
    '''
    def get_num_states(self):
        return (self.agent_1.get_num_states(), self.agent_2.get_num_states())
    
    '''
        Return agent state
    '''
    def get_state(self, obs):
        return (self.agent_1.get_state(obs), self.agent_2.get_state(obs))

    '''
        Return reward
    '''
    def get_reward(self, obs, action):
        return (self.agent_1.get_reward(obs, action), self.agent_2.get_reward(obs, action))
            
    '''
        Return if we must end this episode
    '''
    def get_end(self, obs):
        self.end_1 = self.agent_1.get_end(obs)
        self.end_2 = self.agent_2.get_end(obs)

        return end_1 and end_2

    '''
        Return
    '''
    def check_done(self, obs, last_step):
        return (self.agent_1.check_done(obs, last_step), self.agent_2.check_done(obs, last_step))
      
    '''
        Return function of new action
    '''
    def get_action(self, obs, action):
        return (self.agent_1.get_action(obs, action), self.agent_2.get_action(obs, action))
        
    '''
        Return if current action is available in the environment
    '''
    def check_action_available(self, obs, action, func):
        return (self.agent_1.check_action_available(obs, action, func), self.agent_2.check_action_available(obs, action, func))
        