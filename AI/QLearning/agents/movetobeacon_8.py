import math
import os
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import sys
sys.path.append('./QLearning/')
from qtable import QTable

MAP_NAME = 'MoveToBeacon'
FILE_NAME = 'mtb8'
EPISODES = 500

'''
    Agent class must have this methods:

        class Agent:
            def preprare(obs)
            def step(env, func)
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

class Agent(QTable):

    _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    _PLAYER_SELF = 1
    _PLAYER_NEUTRAL = 3

    _NO_OP = actions.FUNCTIONS.no_op.id
    _MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
    _SELECT_ARMY = actions.FUNCTIONS.select_army.id

    _MOVE_UP = 0
    _MOVE_DOWN = 1
    _MOVE_RIGHT = 2
    _MOVE_LEFT = 3
    _MOVE_UP_RIGHT = 4
    _MOVE_UP_LEFT = 5
    _MOVE_DOWN_RIGHT = 6
    _MOVE_DOWN_LEFT = 7

    _SELECT_ALL = [0]
    _NOT_QUEUED = [0]
    _QUEUED = [1]

    _MOVE_VAL = 3.5

    possible_actions = [
        _MOVE_UP,
        _MOVE_DOWN,
        _MOVE_RIGHT,
        _MOVE_LEFT,
        _MOVE_UP_RIGHT,
        _MOVE_UP_LEFT,
        _MOVE_DOWN_RIGHT,
        _MOVE_DOWN_LEFT
    ]
    
    def __init__(self, load=False):
        QTable.__init__(self, 
                        actions=self.possible_actions, 
                        episodes=EPISODES, 
                        load=load)
        
        if load:
            QTable.load_states(self, filepath=os.getcwd() + '\\QLearning\\saves\\' + FILE_NAME + '_states.npy')
            QTable.load_qtable(self, filepath=os.getcwd() + '\\QLearning\\saves\\' + FILE_NAME + '_qtable.npy')
    
    '''
        Prepare basic parameters.
        Return funcion for select army and initial action: move UP
    '''                    
    def prepare(self, obs, episode):
        QTable.set_epsilon(self, episode=episode)
        
        beacon_new_pos = self.__get_beacon_pos(obs)
        self.beacon_actual_pos = [beacon_new_pos[0], beacon_new_pos[1]]
        self.oldDist = self.__get_dist(obs)

        return actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL]), 0
    
    '''
        Do step of the environment
    '''
    def step(self, env, func):
        obs = env.step(actions=[func])
        return obs, self.get_end(obs[0])

    '''
        Update basic values and train
    '''
    def update(self, obs, delta):
        self.oldDist = self.__get_dist(obs)
    
    '''
        Train agent
    '''
    def train(self, step, current_state, action, reward, new_state, done):
        QTable.learn(self, state=current_state, action=action, reward=reward, state_=new_state)
    
    '''
        Return agent number of actions
    '''
    def get_num_actions(self):
        return self.num_actions
    
    '''
        Return agent number of states
    '''
    def get_num_states(self):
        return self.num_states
    
    '''
        Return agent state
    '''
    def get_state(self, obs):
        state = -1

        marinex, mariney = self.__get_marine_pos(obs)
        beaconx, beacony = self.__get_beacon_pos(obs)

        direction = [beaconx - marinex, beacony - mariney]
        dist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))

        norm = 1 - ((dist - 4) / (55 - 5))
        norm = round(norm,1)

        if norm < 0.1:
            state = 0
        elif norm < 0.2:
            state = 1
        elif norm < 0.3:
            state = 2
        elif norm < 0.4:
            state = 3
        elif norm < 0.5:
            state = 4
        elif norm < 0.6:
            state = 5
        elif norm < 0.7:
            state = 6
        elif norm < 0.8:
            state = 7
        elif norm < 0.9:
            state = 8
        else:
            state = 9
        
        # angle between marine and beacon
        vector_1 = [0, -1]

        np.linalg.norm(direction)

        angleD = self.__ang(vector_1, direction)

        if direction[0] > 0:
            angleD = 360 - angleD
        if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
            state += 0
        elif angleD >= 22.5 and angleD < 67.5:
            state += 1 * 10
        elif angleD >= 67.5 and angleD < 112.5:
            state += 2 * 10
        elif angleD >= 112.5 and angleD < 157.5:
            state += 3 * 10
        elif angleD >= 157.5 and angleD < 202.5:
            state += 4 * 10
        elif angleD >= 202.5 and angleD < 247.5:
            state += 5 * 10
        elif angleD >= 247.5 and angleD < 292.5:
            state += 6 * 10
        elif angleD >= 292.5 and angleD < 337.5:
            state += 7 * 10

        return state

    '''
        Return if we ended
    '''
    def get_end(self, obs):
        return False

    '''
        Return if current action was done
    '''
    def check_done(self, obs, last_step):
        beacon_new_pos = self.__get_beacon_pos(obs)
        
        # if we get beacon or it's the last step
        if self.beacon_actual_pos[0] != beacon_new_pos[0] or self.beacon_actual_pos[1] != beacon_new_pos[1] or last_step:
            return True

        return False

    '''
        Return reward
    '''
    def get_reward(self, obs, action):
        beacon_new_pos = self.__get_beacon_pos(obs)

        if self.beacon_actual_pos[0] != round(beacon_new_pos[0],1) or self.beacon_actual_pos[1] != round(beacon_new_pos[1],1):
            self.beacon_actual_pos = [round(beacon_new_pos[0],1), round(beacon_new_pos[1],1)]
            # get beacon
            reward = 1
        else:
            reward = 0

        return reward

    '''
        Return function of new action
    '''
    def get_action(self, obs, action):
        marinex, mariney = self.__get_marine_pos(obs)
        func = actions.FunctionCall(self._NO_OP, [])

        if  self.possible_actions[action] == self._MOVE_UP:
            if(mariney - self._MOVE_VAL < 3.5):
                mariney += self._MOVE_VAL
            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex, mariney - self._MOVE_VAL]])

        elif self.possible_actions[action] == self._MOVE_UP_LEFT:
            if(marinex  - self._MOVE_VAL < 3.5):
                marinex += self._MOVE_VAL
            if(mariney - self._MOVE_VAL < 3.5):
                mariney +=self._MOVE_VAL
            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex-self._MOVE_VAL/2, mariney - self._MOVE_VAL/2]])

        elif self.possible_actions[action] == self._MOVE_LEFT:
            if(marinex - self._MOVE_VAL < 3.5):
                marinex += self._MOVE_VAL

            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex - self._MOVE_VAL, mariney]])

        elif self.possible_actions[action] == self._MOVE_DOWN_LEFT:
            if(marinex - self._MOVE_VAL < 3.5):
                marinex +=self._MOVE_VAL
            if(mariney + self._MOVE_VAL > 44.5):
                mariney -=self._MOVE_VAL
            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex- self._MOVE_VAL/2, mariney + self._MOVE_VAL/2]])

        elif self.possible_actions[action] == self._MOVE_DOWN:
            if(mariney + self._MOVE_VAL > 44.5):
                mariney -= self._MOVE_VAL

            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex, mariney + self._MOVE_VAL]])

        elif self.possible_actions[action] == self._MOVE_DOWN_RIGHT:
            if(marinex + self._MOVE_VAL > 60.5):
                marinex -= self._MOVE_VAL
            if(mariney + self._MOVE_VAL > 44.5):
                mariney -=self._MOVE_VAL
            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex+ self._MOVE_VAL/2, mariney + self._MOVE_VAL/2]])

        elif self.possible_actions[action] == self._MOVE_RIGHT:
            if(marinex + self._MOVE_VAL > 60.5):
                marinex -= self._MOVE_VAL
            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex + self._MOVE_VAL, mariney]])
        elif self.possible_actions[action] == self._MOVE_UP_RIGHT:
            if(marinex + self._MOVE_VAL > 60.5):
                marinex -= self._MOVE_VAL
            if(mariney - self._MOVE_VAL < 3.5):
                mariney +=self._MOVE_VAL
            func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [marinex+self._MOVE_VAL/2, mariney - self._MOVE_VAL/2]])

        return func

    '''
        Return if current action is available in the environment
    '''
    def check_action_available(self, obs, action, func):
        if not (self._MOVE_SCREEN in obs.observation.available_actions):
            func = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])
        return func

    '''
        (Private method)
        Return marine position
    '''
    def __get_marine_pos(self, obs):
        ai_view = obs.observation['feature_screen'][self._PLAYER_RELATIVE]
        marineys, marinexs = (ai_view == self._PLAYER_SELF).nonzero()
        if len(marinexs) == 0:
            marinexs = np.array([0])
        if len(marineys) == 0:
            marineys = np.array([0])
        marinex, mariney = marinexs.mean(), marineys.mean()
        return marinex, mariney

    '''
        (Private method)
        Return beacon position
    '''
    def __get_beacon_pos(self, obs):
        ai_view = obs.observation['feature_screen'][self._PLAYER_RELATIVE]
        beaconys, beaconxs = (ai_view == self._PLAYER_NEUTRAL).nonzero()
        if len(beaconxs) == 0:
            beaconxs = np.array([0])
        if len(beaconys) == 0:
            beaconys = np.array([0])
        beaconx, beacony = beaconxs.mean(), beaconys.mean()
        return [beaconx, beacony]

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
        Return dist from marine and beacon position
    '''
    def __get_dist(self, obs):
        marinex, mariney = self.__get_marine_pos(obs)
        beaconx, beacony = self.__get_beacon_pos(obs)

        newDist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))
        return newDist