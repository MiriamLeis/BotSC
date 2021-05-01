import numpy as np
import math

from pysc2.lib import actions, features, units

from agents.abstract_base import AbstractBase
from agents.dzwb2v2.internalagent_dq import DQInternalAgent

class DefeatZealots2v2(AbstractBase): 

    TYPE_AGENT_1 = units.Protoss.Stalker
    TYPE_AGENT_2 = units.Zerg.Roach

    def __init__(self):
        super().__init__()

        self.agent_1 = DQInternalAgent(unit_type=self.TYPE_AGENT_1, allies_type=[self.TYPE_AGENT_2])
        self.agent_2 = DQInternalAgent(unit_type=self.TYPE_AGENT_2, allies_type=[self.TYPE_AGENT_2])
    '''
        Return map information.
    '''
    def get_args(self):
        super().get_args()

        return ['DefeatZealotswithBlink_2vs2']
    '''
        Return map information.
    '''
    def get_info(self):
        super().get_args()

        return {'agents':[self.agent_1, self.agent_2]}

    def prepare(self, env):
        return

    def update(self, env, deltaTime):
        return
        
    def step(self, env, evironment):
        return

    def get_state(self, env):
        return

    def get_action(self, env, action):
        return

    def get_reward(self, env, action):
        return

    def get_end(self, env):
        return