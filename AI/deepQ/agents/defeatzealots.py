import numpy as np
import math
import random

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

'''
    Add these parameters mandatory
'''

# network values

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 50  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# environment values

MAP_NAME = 'DefeatZealotswithBlink'
FILE_NAME = 'zealotsModel'


'''
    Agent class must have this methods:

        class Agent:
            def preprare()
            def update()
            def get_num_actions()
            def get_num_states()
            def get_state()
            def get_action()
            def get_reward()
            def check_done()
'''

class Agent:

    _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    _PLAYER_SELF = 1
    _PLAYER_NEUTRAL = 3

    _NO_OP = actions.FUNCTIONS.no_op.id
    _MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
    _ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
    _SELECT_ARMY = actions.FUNCTIONS.select_army.id

    _SELECT_ALL = [0]
    _NOT_QUEUED = [0]
    _QUEUED = [1]

    _MOVE_VAL = 3.5

    _UP = 0
    _DOWN = 1
    _RIGHT = 2
    _LEFT = 3
    _ATTACK = 4

    possible_actions = [
        _UP,
        _DOWN,
        _RIGHT,
        _LEFT,
        _ATTACK
    ]

    '''
        Initialize the agent
    '''
    def __init__(self):
        self.num_actions = len(self.possible_actions)
        self.num_states = 8

    '''
        Prepare basic parameters.
    '''
    def prepare(self, obs):
        return actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL]), 0

    '''
        Update basic values
    '''
    def update(self, obs):
        return
    
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
        return 0

    '''
        Return reward
    '''
    def get_reward(self, obs):
        return 0
    
    '''
        Return if we ended
    '''
    def get_end(self, obs):
        result = self.__get_stalker_position(obs)
        return result[-1]

    '''
        Return if current action was done
    '''
    def check_done(self, obs, last_step):
        return False

    '''
        Return function of new action
    '''
    def get_action(self, obs, action):
        result = self.__get_stalker_position(obs)
        stalkerx = result[0] 
        stalkery = result[1]

        func = actions.FunctionCall(self._NO_OP, [])

        # ATTACK ACTION
        
        # MOVING ACTION
        if self._MOVE_SCREEN in obs.observation.available_actions:

            if  self.possible_actions[action] == self._UP:
                if(stalkery - self._MOVE_VAL < 3.5):
                    stalkery += self._MOVE_VAL
                func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [stalkerx, stalkery - self._MOVE_VAL]])
                marineNextPosition = [stalkerx, stalkery - self._MOVE_VAL]

            elif self.possible_actions[action] == self._DOWN:
                if(stalkery + self._MOVE_VAL > 44.5):
                    stalkery -= self._MOVE_VAL

                func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [stalkerx, stalkery + self._MOVE_VAL]])
                marineNextPosition = [stalkerx, stalkery + self._MOVE_VAL]

            elif self.possible_actions[action] == self._RIGHT:
                if(stalkerx + self._MOVE_VAL > 60.5):
                    stalkerx -= self._MOVE_VAL
                func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [stalkerx + self._MOVE_VAL, stalkery]])
                marineNextPosition = [stalkerx + self._MOVE_VAL, stalkery]

            else:
                if(stalkerx - self._MOVE_VAL < 3.5):
                    stalkerx += self._MOVE_VAL

                func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, [stalkerx - self._MOVE_VAL, stalkery]])
                marineNextPosition = [stalkerx - self._MOVE_VAL, stalkery]

        else:
            func = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])

        return func

    def __get_stalker_position(self, obs):
        stalkers = [unit for unit in obs.observation['feature_units'] 
                    if unit.unit_type == units.Protoss.Stalker]
        if stalkers:
            stalker = random.choice(stalkers)
            return [stalker.y, stalker.x, False]
        else:
            return [-1, -1, True]
