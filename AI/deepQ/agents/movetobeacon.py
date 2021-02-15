import numpy as np
import math

from pysc2.lib import actions
from pysc2.lib import features

# network values

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 50  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# environment values

MAP_NAME = 'MoveToBeacon'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_SELECT_ALL = [0]
_NOT_QUEUED = [0]
_QUEUED = [1]

_MOVE_VAL = 3.5

_UP = 0
_DOWN = 1
_RIGHT = 2
_LEFT = 3

possible_actions = [
    _UP,
    _DOWN,
    _RIGHT,
    _LEFT
]

class Agent:
    '''
        Initialize the agent
    '''
    def __init__(self):
        self.num_actions = len(possible_actions)
        self.num_states = 8

    '''
        Prepare basic parameters.
        Return funcion for select army and initial action: move UP
    '''
    def prepare(self, obs):
        self.beacon_actual_pos = self.__get_beacon_pos(obs)
        self.oldDist = self.__get_dist(obs)

        return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]), 0

    '''
        Update basic values
    '''
    def update(self, obs):
        self.oldDist = self.__get_dist(obs)
    
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
        marinex, mariney = self.__get_marine_pos(obs)
        beaconx, beacony = self.__get_beacon_pos(obs)


        direction = [beaconx-marinex, beacony - mariney]
        dist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))
        
        vector_1 = [0, -1]

        np.linalg.norm(direction)

        angleD = self.__ang(vector_1, direction)

        if direction[0] > 0:
            angleD = 360 - angleD

        state = -1
        if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
            state = [1,0,0,0,0,0,0,0]
        elif angleD >= 22.5 and angleD < 67.5:
            state = [0,1,0,0,0,0,0,0]
        elif angleD >= 67.5 and angleD < 112.5:
            state = [0,0,1,0,0,0,0,0]
        elif angleD >= 112.5 and angleD < 157.5:
            state = [0,0,0,1,0,0,0,0]
        elif angleD >= 157.5 and angleD < 202.5:
            state = [0,0,0,0,1,0,0,0]
        elif angleD >= 202.5 and angleD < 247.5:
            state = [0,0,0,0,0,1,0,0]
        elif angleD >= 247.5 and angleD < 292.5:
            state = [0,0,0,0,0,0,1,0]
        elif angleD >= 292.5 and angleD < 337.5:
            state = [0,0,0,0,0,0,0,1]

        return state


    '''
        Return if current action was done
    '''
    def check_done(self, obs, last_step):
        beacon_new_pos = self.__get_beacon_pos(obs)
        
        # if we get beacon or it's the last step
        if self.beacon_actual_pos[0] != beacon_new_pos[0] or self.beacon_actual_pos[1] != beacon_new_pos[1] or last_step:
            self.beacon_actual_pos = [beacon_new_pos[0], beacon_new_pos[1]]
            return True

        return False

    '''
        Return function of new action
    '''
    def get_action(self, obs, action):
        marinex, mariney = self.__get_marine_pos(obs)
        func = actions.FunctionCall(_NO_OP, [])
        
        if  possible_actions[action] == _UP:
            if(mariney - _MOVE_VAL < 3.5):
                mariney += _MOVE_VAL
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex, mariney - _MOVE_VAL]])
            marineNextPosition = [marinex, mariney - _MOVE_VAL]

        elif possible_actions[action] == _DOWN:
            if(mariney + _MOVE_VAL > 44.5):
                mariney -= _MOVE_VAL

            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex, mariney + _MOVE_VAL]])
            marineNextPosition = [marinex, mariney + _MOVE_VAL]

        elif possible_actions[action] == _RIGHT:
            if(marinex + _MOVE_VAL > 60.5):
                marinex -= _MOVE_VAL
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex + _MOVE_VAL, mariney]])
            marineNextPosition = [marinex + _MOVE_VAL, mariney]

        else:
            if(marinex - _MOVE_VAL < 3.5):
                marinex += _MOVE_VAL

            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex - _MOVE_VAL, mariney]])
            marineNextPosition = [marinex - _MOVE_VAL, mariney]

        return func

    '''
        Return reward
    '''
    def get_reward(self, obs):
        return self.oldDist - self.__get_dist(obs)
    
    '''
        (Private method)
        Return marine position
    '''
    def __get_marine_pos(self, obs):
        ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
        marineys, marinexs = (ai_view == _PLAYER_SELF).nonzero()
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
        ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
        beaconys, beaconxs = (ai_view == _PLAYER_NEUTRAL).nonzero()
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

