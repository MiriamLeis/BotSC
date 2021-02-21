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

    _MOVE_VAL = 3.5
    _RADIO_VAL = 15

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
        self.enemy_totalHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        self.ally_totalHP = self.__get_group_totalHP(obs, units.Protoss.Stalker)
        self.last_dist = self.__get_dist(self.__get_stalker(obs), self.__get_zealot(obs))

        return actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL]), 0

    '''
        Update basic values
    '''
    def update(self, obs, delta):
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
        stalker = self.__get_stalker(obs)
        zealot = self.__get_zealot(obs)

        # get direction
        direction = [zealot.x - stalker.x, zealot.y - stalker.y]
        np.linalg.norm(direction)

        vector_1 = [0, -1]
        angleD = self.__ang(vector_1, direction)

        if direction[0] > 0:
            angleD = 360 - angleD
        
        # check proximity
        dist = self.__get_dist(stalker, zealot)
        if dist <= self._RADIO_VAL: 
            dist = 1
        else: 
            dist = 0

        # check if i can shoot
        can_shoot = 0
        if stalker.weapon_cooldown == 0:
            can_shoot = 1

        # prepare state
        state = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

        if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
            state[0] = [1, can_shoot, dist]
        elif angleD >= 22.5 and angleD < 67.5:
            state[1] = [1, can_shoot, dist]
        elif angleD >= 67.5 and angleD < 112.5:
            state[2] = [1, can_shoot, dist]
        elif angleD >= 112.5 and angleD < 157.5:
            state[3] = [1, can_shoot, dist]
        elif angleD >= 157.5 and angleD < 202.5:
            state[4] = [1, can_shoot, dist]
        elif angleD >= 202.5 and angleD < 247.5:
            state[5] = [1, can_shoot, dist]
        elif angleD >= 247.5 and angleD < 292.5:
            state[6] = [1, can_shoot, dist]
        elif angleD >= 292.5 and angleD < 337.5:
            state[7] = [1, can_shoot, dist]

        return state

    '''
        Return reward
    '''
    def get_reward(self, obs):
        reward = 0

        # reward for moving
        dist = self.__get_dist(self.__get_stalker(obs), self.__get_zealot(obs))
        if dist > self._RADIO_VAL:
            reward += self.last_dist - dist

        # reward for attacking
        actual_enemy_totalHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        actual_ally_totalHP = self.__get_group_totalHP(obs, units.Protoss.Stalker)

        if actual_enemy_totalHP > self.enemy_totalHP:
            reward += 10
        
        else:
            reward += (self.enemy_totalHP - actual_enemy_totalHP) - (self.ally_totalHP - actual_ally_totalHP)
        
        self.enemy_totalHP = actual_enemy_totalHP
        self.ally_totalHP = actual_ally_totalHP

        return reward
    
    '''
        Return if we must end this episode
    '''
    def get_end(self, obs):
        stalkers = self.__get_group(obs, units.Protoss.Stalker)
        return not stalkers

    '''
        Return if current action was done
    '''
    def check_done(self, obs, last_step):
        return False

    '''
        Return function of new action
    '''
    def get_action(self, obs, action):
        func = actions.FunctionCall(self._NO_OP, [])

        if self.possible_actions[action] == self._ATTACK:
            # ATTACK ACTION
            if self._ATTACK_SCREEN in obs.observation.available_actions:
                zealot = self.__get_zealot(obs)
                func = actions.FunctionCall(self._ATTACK_SCREEN, [self._NOT_QUEUED, [zealot.x, zealot.y]])

        else:
            # MOVING ACTION
            if self._MOVE_SCREEN in obs.observation.available_actions:
                stalker = self.__get_stalker(obs)

                if  self.possible_actions[action] == self._UP:
                    if(stalker.y - self._MOVE_VAL < 3.5):
                        stalker.y += self._MOVE_VAL
                    marineNextPosition = [stalker.x, stalker.y - self._MOVE_VAL]

                elif self.possible_actions[action] == self._DOWN:
                    if(stalker.y + self._MOVE_VAL > 44.5):
                        stalker.y -= self._MOVE_VAL
                    marineNextPosition = [stalker.x, stalker.y + self._MOVE_VAL]

                elif self.possible_actions[action] == self._RIGHT:
                    if(stalker.x + self._MOVE_VAL > 60.5):
                        stalker.x -= self._MOVE_VAL
                    marineNextPosition = [stalker.x + self._MOVE_VAL, stalker.y]

                else:
                    if(stalker.x - self._MOVE_VAL < 3.5):
                        stalker.x += self._MOVE_VAL
                    marineNextPosition = [stalker.x - self._MOVE_VAL, stalker.y]

                func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, marineNextPosition])

            # SELECT ARMY
            else:
                func = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])

        return func

    def __get_group(self, obs, group_type):
        group = [unit for unit in obs.observation['feature_units'] 
                    if unit.unit_type == group_type]
        return group

    def __get_stalker(self, obs):
        stalkers = self.__get_group(obs, units.Protoss.Stalker)
        return random.choice(stalkers)


    def __get_zealot(self, obs):
        zealots = self.__get_group(obs, units.Protoss.Zealot)

        # search who has lower hp and lower shield
        target = zealots[0]
        for i in range(1, len(zealots)):
            if zealots[i].health < target.health or (zealots[i].health == target.health and zealots[i].shield < target.shield) :
                target = zealots[i]
                
        return target

    def __get_group_totalHP(self, obs, group_type):
        group = self.__get_group(obs, group_type)
        totalHP = 0
        for unit in group:
            totalHP += unit.health
        return totalHP

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
        Return dist from stalker and zealot position
    '''
    def __get_dist(self, stalker, zealot):
        newDist = math.sqrt(pow(stalker.x - zealot.x, 2) + pow(stalker.y - zealot.y, 2))
        return newDist

