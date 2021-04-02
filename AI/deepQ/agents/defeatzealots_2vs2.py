import numpy as np
import math
import random
import os

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

from deepQ.agents.defeatzealots_2enemies import Agent as internal_agent

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

        # get units
        group = [unit for unit in obs.observation['feature_units'] 
                    if unit.unit_type == units.Protoss.Stalker]
        self.unit_1 = group[0]
        self.unit_2 = group[1]

        # first action is trash
        return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]]), (state_1, state_2)

    '''
        Do step of the environment
    '''
    def step(self, env, func):
        if not self.end_1:
            obs = env.step(actions=[self.select_1])
            self.end_1 = self.agent_1.get_end(obs[0])

            if not self.end_1:
                obs, self.end_1 = self.agent_1.step(env, func[0])

        self.end_2 = self.agent_2.get_end(obs[0])
        if not self.end_2:
            obs = env.step(actions=[self.select_2])
            self.end_2 = self.agent_2.get_end(obs[0])

            if not self.end_2:
                obs, self.end_2 = self.agent_2.step(env, func[1])
                self.end_1 = self.agent_1.get_end(obs[0])

        return obs, self.end_1 and self.end_2

    '''
        Update basic values
    '''
    def update(self, obs, delta):
        self.agent_1.update(obs, delta)
        self.agent_2.update(obs, delta)

        self.select_1 = actions.FUNCTIONS.select_point("select",(self.unit_1[1], self.unit_1[2]))
        self.select_2 = actions.FUNCTIONS.select_point("select",(self.unit_2[1], self.unit_2[2]))
    
    '''
        Train agents (tuple)
    '''
    def train(self, step, current_state, action, reward, new_state, done):
        self.agent_1.train(step, current_state[0], action[0], reward[0], new_state[0], done[0])
        self.agent_2.train(step, current_state[1], action[1], reward[1], new_state[1], done[1])
        
    '''
        Return actions choosen by our agents (tuple)
    '''
    def choose_action(self, current_state):
        return (self.agent_1.choose_action(current_state[0]), self.agent_2.choose_action(current_state[1]))

    '''
        Return agents number of actions (tuple)
    '''
    def get_num_actions(self):
        return (self.agent_1.get_num_actions(), self.agent_2.get_num_actions())
    
    '''
        Return agents number of states (tuple)
    '''
    def get_num_states(self):
        return (self.agent_1.get_num_states(), self.agent_2.get_num_states())
    
    '''
        Return agents states (tuple)
    '''
    def get_state(self, obs):
        return (self.agent_1.get_state(obs), self.agent_2.get_state(obs))

    '''
        Return rewards (tuple)
    '''
    def get_reward(self, obs, action):
        return (self.agent_1.get_reward(obs, action[0]), self.agent_2.get_reward(obs, action[1]))
            
    '''
        Return if we must end this episode
    '''
    def get_end(self, obs):
        self.end_1 = self.agent_1.get_end(obs)
        self.end_2 = self.agent_2.get_end(obs)

        return self.end_1 and self.end_2

    '''
        Return
    '''
    def check_done(self, obs, last_step):
        return (self.agent_1.check_done(obs, last_step), self.agent_2.check_done(obs, last_step))
      
    '''
        Return function of new action
    '''
    def get_action(self, obs, action):
        return (self.agent_1.get_action(obs, action[0]), self.agent_2.get_action(obs, action[1]))
        
    '''
        Return if current action is available in the environment
    '''
    def check_action_available(self, obs, action, func):
        return (self.agent_1.check_action_available(obs, action[0], func[0]), self.agent_2.check_action_available(obs, action[1], func[1]))
        