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

MAP_NAME = 'DefeatZealotswithBlink_2vs2'
FILE_NAME = 'zealots2vs2Model'
EPISODES = 1000


'''
    Agent class must have this methods:

        class Agent:
            def save(filepath)
            def prepare(obs)
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

    TYPE_AGENT_1 = units.Protoss.Stalker
    TYPE_AGENT_2 = units.Terran.Ghost

    '''
        Initialize the agent
    '''
    def __init__(self, load=False):
        self.end_1 = False
        self.end_2 = False

        self.agent_1 = internal_agent(num_states=21, unit_type=self.TYPE_AGENT_1)
        self.agent_2 = internal_agent(num_states=21, unit_type=self.TYPE_AGENT_2)
        
        if load:
            self.agent_1.loadModel(os.getcwd() + '\\deepQ\\models\\' + FILE_NAME + '_1.h5')
            self.agent_2.loadModel(os.getcwd() + '\\deepQ\\models\\' + FILE_NAME + '_2.h5')
    
    def save(self, filepath):
        self.agent_1.save(filepath + '_1')
        self.agent_2.save(filepath + '_2')

    '''
        Prepare basic parameters. This is called before start the episode.
    '''
    def prepare(self, obs, episode):
        self.end_1 = False
        self.end_2 = False

        _, state_1 = self.agent_1.prepare(obs, episode)
        _, state_2 = self.agent_2.prepare(obs, episode)

        # get units
        self.__set_units(obs)

        # first action is trash
        return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]]), (state_1, state_2)

    '''
        Do step of the environment
    '''
    def step(self, env, func):
        # agent1 steps
        if not self.end_1:
            obs = env.step(actions=[self.select_1])
            self.end_1 = self.agent_1.get_end(obs[0])

            if not self.end_1:
                obs, self.end_1 = self.agent_1.step(env, func[0])
                
            self.end_2 = self.agent_2.get_end(obs[0])

        # agent2 steps
        if not self.end_2:
            obs = env.step(actions=[self.select_2])
            self.end_2 = self.agent_2.get_end(obs[0])
            
            if not self.end_2:
                obs, self.end_2 = self.agent_2.step(env, func[1])

        # check if agent1 died in these steps
        self.end_1 = self.agent_1.get_end(obs[0])

        return obs, self.end_1 and self.end_2

    '''
        Update basic values
    '''
    def update(self, obs, delta):
        if not self.end_1:
            self.agent_1.update(obs, delta)
        if not self.end_2:
            self.agent_2.update(obs, delta)

        # get units
        self.__set_units(obs)

        # set select actions
        if not self.end_1:
            self.select_1 = actions.FUNCTIONS.select_point("select",(self.unit_1[1], self.unit_1[2]))
        if not self.end_2:
            self.select_2 = actions.FUNCTIONS.select_point("select",(self.unit_2[1], self.unit_2[2]))
    
    '''
        Train agents (tuple)
    '''
    def train(self, step, current_state, action, reward, new_state, done):
        if not self.end_1:
            self.agent_1.train(step, current_state[0], action[0], reward[0], new_state[0], done[0])
        if not self.end_2:
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
        # get original states
        state_1 = self.agent_1.get_state(obs)
        state_2 = self.agent_2.get_state(obs)

        new_state_1 = [0,0,0,0,0,0,0,0]
        new_state_2 = [0,0,0,0,0,0,0,0]
        
        # check direction between agents if both are alive
        if not self.end_1 and not self.end_2:
            direction = [self.unit_2.x - self.unit_1.x, self.unit_2.y - self.unit_1.y]
            np.linalg.norm(direction)

            vector_1 = [0, -1]
            angleD = self.__ang(vector_1, direction)

            if direction[0] > 0:
                angleD = 360 - angleD

            # check dist
            dist = self.__get_dist([self.unit_1.x, self.unit_1.y], [self.unit_2.x, self.unit_2.y])

            norm = 1 - ((dist - 4) / (55 - 5))
            norm = round(norm, 1)

            # check angle between agents. agent2 have opposite state value than agent1
            if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
                new_state_1[0] = norm
                new_state_2[4] = norm
            elif angleD >= 22.5 and angleD < 67.5:
                new_state_1[1] = norm
                new_state_2[5] = norm
            elif angleD >= 67.5 and angleD < 112.5:
                new_state_1[2] = norm
                new_state_2[6] = norm
            elif angleD >= 112.5 and angleD < 157.5:
                new_state_1[3] = norm
                new_state_2[7] = norm
            elif angleD >= 157.5 and angleD < 202.5:
                new_state_1[4] = norm
                new_state_2[0] = norm
            elif angleD >= 202.5 and angleD < 247.5:
                new_state_1[5] = norm
                new_state_2[1] = norm
            elif angleD >= 247.5 and angleD < 292.5:
                new_state_1[6] = norm
                new_state_2[2] = norm
            elif angleD >= 292.5 and angleD < 337.5:
                new_state_1[7] = norm
                new_state_2[3] = norm

        # add agent2 position information to agent1
        state_1 += new_state_1

        # add agent1 position information to agent2
        state_2 += new_state_2

        return (state_1, state_2)

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

    '''
        (Private method)
        Return angle formed from two lines
    '''
    def __set_units(self, obs):
        # get units
        group = [unit for unit in obs.observation['feature_units'] 
                    if unit.unit_type == self.TYPE_AGENT_1]
        if group:
            self.unit_1 = group[0]

        # get units
        group = [unit for unit in obs.observation['feature_units'] 
                    if unit.unit_type == self.TYPE_AGENT_2]
        if group:
            self.unit_2 = group[0]


    '''
        (Private method)
        Return angle formed from two lines
    '''
    def __ang(self, lineA, lineB):
        # Get nicer vector form
        vA = lineA
        vB = lineB
        # Get dot prod
        dot_prod = np.dot(vA, vB)
        # Get magnitudes
        magA = np.dot(vA, vA)**0.5
        magB = np.dot(vB, vB)**0.5
        # Get cosine value
        cos = dot_prod/magA/magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod/magB/magA)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle)%360


        return ang_deg

    '''
        (Private method)
        Return dist between A and B
    '''
    def __get_dist(self, A, B):
        newDist = math.sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2))
        return newDist
        