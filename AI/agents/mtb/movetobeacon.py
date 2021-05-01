import numpy as np
import math

from pysc2.lib import actions, features

from agents.abstract_base import AbstractBase

class MoveToBeacon(AbstractBase): 
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

    def __init__(self):
        super().__init__()
    
    '''
        Return map information.
    '''
    def get_args(self):
        super().get_args()

        return ['MoveToBeacon']

    '''
        Prepare basic parameters.
    '''
    def prepare(self, env):
        super().prepare(env=env)

        beacon_new_pos = self._get_unit_pos(env=env,view=self._PLAYER_NEUTRAL)
        self.beacon_actual_pos = [beacon_new_pos[0], beacon_new_pos[1]]

        self.oldDist = self._get_dist(env)

        self.action = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])

        return 0

    '''
        Update basic values and train
    '''
    def update(self, env, deltaTime):
        super().update(env=env, deltaTime=deltaTime)

        self.oldDist = self._get_dist(env)

    '''
        Do step of the environment
    '''
    def step(self, env, environment):
        self._check_action_available(env=env)
        obs = environment.step(actions=[self.action])
        return obs, self.get_end(env=env)

    '''
        Return action of environment
    '''
    def get_action(self, env, action):
        marinex, mariney = self._get_unit_pos(env=env, view=self._PLAYER_SELF)
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
        
        self.action = func
    
    '''
        Return reward
    '''
    def get_reward(self, env, action):
        beacon_new_pos = self._get_unit_pos(env=env, view=self._PLAYER_NEUTRAL)

        if self.beacon_actual_pos[0] != round(beacon_new_pos[0],1) or self.beacon_actual_pos[1] != round(beacon_new_pos[1],1):
            self.beacon_actual_pos = [round(beacon_new_pos[0],1), round(beacon_new_pos[1],1)]
            # get beacon
            reward = 1
        else:
            reward = 0

        return reward

    '''
        Return True if we must end this episode
    '''
    def get_end(self, env):
        return False

    '''
        (Protected method)
        Check if current action is available. If not, use default action
    '''
    def _check_action_available(self, env):
        if not (self._MOVE_SCREEN in env.observation.available_actions):
            self.action = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])

    '''
        (Protected method)
        Return unit position
    '''
    def _get_unit_pos(self, env, view):
        ai_view = env.observation['feature_screen'][self._PLAYER_RELATIVE]
        unitys, unitxs = (ai_view == view).nonzero()
        if len(unitxs) == 0:
            unitxs = np.array([0])
        if len(unitys) == 0:
            unitys = np.array([0])
        return unitxs.mean(), unitys.mean()

    '''
        (Protected method)
        Return angle formed from two lines
    '''
    def _ang(self, lineA, lineB):
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
        (Protected method)
        Return dist from marine and beacon position
    '''
    def _get_dist(self, env):
        marinex, mariney = self._get_unit_pos(env, self._PLAYER_SELF)
        beaconx, beacony = self._get_unit_pos(env, self._PLAYER_NEUTRAL)

        newDist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))
        return newDist