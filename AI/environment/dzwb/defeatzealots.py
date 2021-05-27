import numpy as np
import math

from pysc2.lib import actions, features, units

from environment.starcraft_env import StarCraftEnv # environment

class DefeatZealots(StarCraftEnv): 
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

    UNIT_ALLY = units.Protoss.Stalker
    UNIT_ENEMY = units.Protoss.Zealot

    def __init__(self,args):
        super().__init__(args=args)

    '''
        Prepare basic parameters. This is called before start the episode.
    '''
    def prepare(self):
        self.enemy_totalHP = self._get_group_totalHP(group_type=self.UNIT_ENEMY)
        self.enemy_originalHP = self.enemy_totalHP
        self.enemy_onlyHP = self._get_group_totalHP(group_type=self.UNIT_ENEMY)
        self.ally_totalHP = self._get_group_totalHP(group_type=self.UNIT_ALLY)
        self.ally_originalHP = self.ally_totalHP
        self.last_dist = self._get_dist(self._get_meangroup_position(group_type=self.UNIT_ALLY), self._get_meangroup_position(group_type=self.UNIT_ENEMY))

        self.last_can_shoot = False
        self.dead = False

        self.action = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])

    '''
        Update values
    '''
    def update(self, deltaTime):
        self.last_can_shoot = self.current_can_shoot
        
    '''
        Do step of the environment
    '''
    def step(self):
        self.__check_action_available()
        obs = self.env.step(actions=[self.action])
        self.obs = obs[0]

    '''
        Return action of environment
    '''
    def get_action(self, action):
        func = actions.FunctionCall(self._NO_OP, [])

        if self.possible_actions[action] == self._ATTACK:
            # ATTACK ACTION
            if self._ATTACK_SCREEN in self.obs.observation.available_actions:
                zealot = self._get_zealot()
                func = actions.FunctionCall(self._ATTACK_SCREEN, [self._NOT_QUEUED, [zealot.x, zealot.y]])

        else:
            # MOVING ACTION
            if self._MOVE_SCREEN in self.obs.observation.available_actions:
                stalkerx, stalkery = self._get_meangroup_position(group_type=self.UNIT_ALLY)

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

        self.num_action = action
        self.action = func

    '''
        Return reward
    '''
    def get_reward(self, action):
        reward = 0

        # reward for attacking
        actual_enemy_totalHP = self._get_group_totalHP(group_type=self.UNIT_ENEMY)
        actual_ally_totalHP = self._get_group_totalHP(group_type=self.UNIT_ALLY)
        actual_enemy_onlyHP = self._get_group_onlyHP(group_type=self.UNIT_ENEMY)

        diff = (self.enemy_totalHP - actual_enemy_totalHP) - (self.ally_totalHP - actual_ally_totalHP)

        # check if we made some damage and we have shot with this action
        if actual_enemy_totalHP < self.enemy_totalHP and action == self._ATTACK:
            reward += 1
        
        if actual_ally_totalHP < self.ally_totalHP:
            reward += -1
            self.dead = False

        #update values
        self.enemy_totalHP = actual_enemy_totalHP
        self.enemy_onlyHP = actual_enemy_onlyHP
        self.ally_totalHP = actual_ally_totalHP

        return reward

    '''
        Return True if we must end this episode
    '''
    def get_end(self):
        stalkers = self._get_group(group_type=self.UNIT_ALLY)
        self.dead = not stalkers
        return self.dead

    '''
        (Protected method)
        Return specified group
    '''
    def _get_group(self, group_type):
        group = [unit for unit in self.obs.observation['feature_units'] 
                    if unit.unit_type == group_type]
        return group

    '''
        (Protected method)
        Return zealot with lowest healt (health + shield)
    '''
    def _get_zealot(self):
        zealots = self._get_group(group_type=self.UNIT_ENEMY)

        # search who has lower hp and lower shield
        target = zealots[0]
        for i in range(1, len(zealots)):
            if zealots[i].health < target.health or (zealots[i].health == target.health and zealots[i].shield < target.shield) :
                target = zealots[i]
                
        return target

    '''
        (Protected method)
        Return totalHP of a group = (unit health plus unit shield)
    '''

    def _get_group_totalHP(self, group_type):
        group = self._get_group(group_type)
        totalHP = 0
        for unit in group:
            totalHP += unit.health + unit.shield
        return totalHP

    '''
        (Protected method)
        Return totalHP of a group = (unit health plus unit shield)
    '''

    def _get_group_onlyHP(self, group_type):
        group = self._get_group(group_type)
        totalHP = 0
        for unit in group:
            totalHP += unit.health
        return totalHP

    '''
        (Protected method)
        Return mean position of a group
    '''
    def _get_meangroup_position(self, group_type):
        group = self._get_group(group_type)

        unitx = unity = 0
        for unit in group:
            unitx += unit.x
            unity += unit.y
        if len(group) < 0 or len(group) == 0:
                       
            unitx /= 1
            unity /= 1

        else:
            unitx /= len(group)
            unity /= len(group)
        return unitx, unity

    '''
        (Protected method)
        Return True if group cooldown == 0
    '''
    def _can_shoot(self, group_type):
        group = self._get_group(group_type)

        for unit in group:
            if unit.weapon_cooldown != 0:
                return False

        return True

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
        Return dist between A and B
    '''
    def _get_dist(self, A, B):
        newDist = math.sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2))
        return newDist

    '''
        (Private method)
        Return if current action is available in the environment
    '''
    def __check_action_available(self):
        if self.possible_actions[self.num_action] == self._ATTACK:
            # ATTACK ACTION
            if not (self._ATTACK_SCREEN in self.obs.observation.available_actions):
                self.action = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])
        else:
            if not (self._MOVE_SCREEN in self.obs.observation.available_actions):
                self.action = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])