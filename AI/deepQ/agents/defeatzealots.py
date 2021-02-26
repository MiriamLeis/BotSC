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
MIN_REPLAY_MEMORY_SIZE = 200  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
MINIBATCH_SIZE = 200
MAX_CASES = 1000
HIDDEN_NODES = 100
HIDDEN_LAYERS = 2
CASES_TO_DELETE = 100

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
            def get_end()
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

    _MOVE_VAL = 5.5
    _RADIO_VAL = 20

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

    '''
        Initialize the agent
    '''
    def __init__(self):
        self.num_actions = len(self.possible_actions)
        self.num_states = 14

    '''
        Prepare basic parameters.
    '''
    def prepare(self, obs):
        self.enemy_totalHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        self.enemy_onlyHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        self.ally_totalHP = self.__get_group_totalHP(obs, units.Protoss.Stalker)
        self.last_dist = self.__get_dist(self.__get_meangroup_position(obs, units.Protoss.Stalker), self.__get_meangroup_position(obs, units.Protoss.Zealot))
        self.current_can_shoot = False
        self.last_can_shoot = False
        self.current_on_range = False
        self.last_on_range = False

        return actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL]), 0

    '''
        Update basic values
    '''
    def update(self, obs, delta):
        self.last_can_shoot = self.current_can_shoot
        self.last_on_range = self.current_on_range
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
        stalkerx, stalkery = self.__get_meangroup_position(obs, units.Protoss.Stalker)
        zealotx, zealoty = self.__get_meangroup_position(obs, units.Protoss.Zealot)

        # get direction
        direction = [zealotx- stalkerx, zealoty- stalkery]
        np.linalg.norm(direction)

        vector_1 = [0, -1]
        angleD = self.__ang(vector_1, direction)

        if direction[0] > 0:
            angleD = 360 - angleD
        
        # prepare state
        state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # check angle
        if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
            state[0] = 1
        elif angleD >= 22.5 and angleD < 67.5:
            state[1] = 1
        elif angleD >= 67.5 and angleD < 112.5:
            state[2] = 1
        elif angleD >= 112.5 and angleD < 157.5:
            state[3] = 1
        elif angleD >= 157.5 and angleD < 202.5:
            state[4] = 1
        elif angleD >= 202.5 and angleD < 247.5:
            state[5] = 1
        elif angleD >= 247.5 and angleD < 292.5:
            state[6] = 1
        elif angleD >= 292.5 and angleD < 337.5:
            state[7] = 1

        # check cooldown
        if self.__can_shoot(obs, units.Protoss.Stalker):
            state[8] = 1
            self.current_can_shoot = True
        else:
            self.current_can_shoot = False
        
        # check dist
        if self.__get_dist([stalkerx, stalkery], [zealotx, zealoty]) <= self._RADIO_VAL:
            state[9] = 1
            self.current_on_range = True
        else:
            self.current_on_range = False

        if (stalkery - self._MOVE_VAL) < 3.5:
            state[10] = 1
        if (stalkerx - self._MOVE_VAL) < 3.5:
            state[11] = 1
        if (stalkery + self._MOVE_VAL) > 44.5:
            state[12] = 1
        if (stalkerx + self._MOVE_VAL) > 60.5:
            state[13] = 1
        
        return state

    '''
        Return reward
    '''
    def get_reward(self, obs, action):
        reward = 0

        # reward for moving
        stalker = self.__get_stalker(obs)
        dist = self.__get_dist(self.__get_meangroup_position(obs, units.Protoss.Stalker), self.__get_meangroup_position(obs, units.Protoss.Zealot))

            # check if we arent on range but we can shot
        if not self.last_on_range and self.last_can_shoot:
            
                # reward for getting close
            if ((self.last_dist - dist) > 0):
                reward += 1

                # punishment for running away
            else: 
                reward -= 1

            # check if we are on range
        elif self.last_on_range:

                # punishment for getting close
            if ((self.last_dist - dist) >= 0):
                reward -= 1

                # reward for running away
            elif not self.last_can_shoot:
                reward += 2

        # reward for attacking
        actual_enemy_totalHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        actual_ally_totalHP = self.__get_group_totalHP(obs, units.Protoss.Stalker)
        actual_enemy_onlyHP = self.__get_group_onlyHP(obs, units.Protoss.Zealot)

            # check if zealots reapered
        diffOnlyHP = (self.enemy_onlyHP - actual_enemy_onlyHP)
        if diffOnlyHP < 0:
            reward = 2
            return reward

        diff = (self.enemy_totalHP - actual_enemy_totalHP) - (self.ally_totalHP - actual_ally_totalHP)

            # check if we made some damage and we have shot with this action
        if diff > -5 and (action == 8) and self.last_can_shoot:
            reward += 2
        elif diff < 0:
            reward -= 1

        #update values
        self.enemy_totalHP = actual_enemy_totalHP
        self.ally_totalHP = actual_ally_totalHP
        self.last_dist = dist

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
                stalkerx, stalkery = self.__get_meangroup_position(obs, units.Protoss.Stalker)

                if  self.possible_actions[action] == self._UP:
                    if(stalkery - self._MOVE_VAL < 3.5):
                        stalkery += self._MOVE_VAL
                    nextPosition = [stalkerx, stalkery - self._MOVE_VAL]

                elif self.possible_actions[action] == self._UP_LEFT:
                    if(stalkery - self._MOVE_VAL < 3.5):
                        stalkery += self._MOVE_VAL
                    if(stalkerx - self._MOVE_VAL < 3.5):
                        stalkerx += self._MOVE_VAL
                    nextPosition = [stalkerx - (self._MOVE_VAL/2), stalkery - (self._MOVE_VAL/2)]

                elif self.possible_actions[action] == self._LEFT:
                    if(stalkerx - self._MOVE_VAL < 3.5):
                        stalkerx += self._MOVE_VAL
                    nextPosition = [stalkerx - self._MOVE_VAL, stalkery]

                elif self.possible_actions[action] == self._DOWN_LEFT:
                    if(stalkery + self._MOVE_VAL > 44.5):
                        stalkery -= self._MOVE_VAL
                    if(stalkerx - self._MOVE_VAL < 3.5):
                        stalkerx += self._MOVE_VAL
                    nextPosition = [stalkerx - (self._MOVE_VAL/2), stalkery + (self._MOVE_VAL/2)]

                elif self.possible_actions[action] == self._DOWN:
                    if(stalkery + self._MOVE_VAL > 44.5):
                        stalkery -= self._MOVE_VAL
                    nextPosition = [stalkerx, stalkery + self._MOVE_VAL]

                elif self.possible_actions[action] == self._DOWN_RIGHT:
                    if(stalkery + self._MOVE_VAL > 44.5):
                        stalkery -= self._MOVE_VAL
                    if(stalkerx + self._MOVE_VAL > 60.5):
                        stalkerx -= self._MOVE_VAL
                    nextPosition = [stalkerx + (self._MOVE_VAL/2), stalkery + (self._MOVE_VAL/2)]

                elif self.possible_actions[action] == self._RIGHT:
                    if(stalkerx + self._MOVE_VAL > 60.5):
                        stalkerx -= self._MOVE_VAL
                    nextPosition = [stalkerx + self._MOVE_VAL, stalkery]

                elif self.possible_actions[action] == self._UP_RIGHT:
                    if(stalkery - self._MOVE_VAL < 3.5):
                        stalkery += self._MOVE_VAL
                    if(stalkerx + self._MOVE_VAL > 60.5):
                        stalkerx -= self._MOVE_VAL
                    nextPosition = [stalkerx + (self._MOVE_VAL/2), stalkery - (self._MOVE_VAL/2)]

                func = actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, nextPosition])

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

    '''
        (Private method)
        Return totalHP of a group = (unit health plus unit shield)
    '''

    def __get_group_totalHP(self, obs, group_type):
        group = self.__get_group(obs, group_type)
        totalHP = 0
        for unit in group:
            totalHP += unit.health + unit.shield
        return totalHP

    '''
        (Private method)
        Return totalHP of a group = (unit health plus unit shield)
    '''

    def __get_group_onlyHP(self, obs, group_type):
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
        Return mean position of a group
    '''
    def __get_meangroup_position(self, obs, group_type):
        group = self.__get_group(obs, group_type)

        unitx = unity = 0
        for unit in group:
            unitx += unit.x
            unity += unit.y
        
        unitx /= len(group)
        unity /= len(group)

        return unitx, unity

    '''
        (Private method)
        Return True if group cooldown == 0
    '''
    def __can_shoot(self, obs, group_type):
        group = self.__get_group(obs, group_type)

        for unit in group:
            if unit.weapon_cooldown != 0:
                return False

        return True

    '''
        (Private method)
        Return dist between A and B
    '''
    def __get_dist(self, A, B):
        newDist = math.sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2))
        return newDist

