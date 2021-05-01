import numpy as np
import math

from pysc2.lib import actions, features, units

from agents.abstract_base import AbstractBase
from agents.dzwb.internalagent_dq import DQInternalAgent

class DefeatZealots2v2(AbstractBase): 
    '''
        Useful variables 
    '''

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

    _MOVE_VAL = 5.5
    _RADIO_VAL = 20
    _RANGE_VAL = 5

    _UP = 0
    _UP_LEFT = 1
    _LEFT = 2
    _DOWN_LEFT = 3
    _DOWN = 4
    _DOWN_RIGHT = 5
    _RIGHT = 6
    _UP_RIGHT = 7
    _ATTACK = 8

    possible_actions = [
        _UP,
        _UP_LEFT,
        _LEFT,
        _DOWN_LEFT,
        _DOWN,
        _DOWN_RIGHT,
        _RIGHT,
        _UP_RIGHT,
        _ATTACK
    ]

    def __init__(self):
        super().__init__()

        self.agent_1 = DQInternalAgent(unit_type=self.TYPE_AGENT_1)
        self.agent_2 = DQInternalAgent(unit_type=self.TYPE_AGENT_2)
    '''
        Return map information.
    '''
    def get_args(self):
        super().get_args()

        return ['DefeatZealotswithBlink_2enemies']