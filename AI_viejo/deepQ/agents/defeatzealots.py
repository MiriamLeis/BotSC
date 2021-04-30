import numpy as np
import math
import random
import os

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

'''
    Import Neural Network
'''
import sys
sys.path.append('./deepQ/')
from dq_network import DQNAgent

'''
    Add these parameters mandatory
'''

# environment values

MAP_NAME = 'DefeatZealotswithBlink'
FILE_NAME = 'zealotsModel'
EPISODES = 1000


'''
    Agent class must have this methods:

        class Agent:
            def preprare(obs)
            def step(env, func)
            def update(obs, deltaTime)
            def train(step, current_state, action, reward, new_state, done)
            def get_state(obs)
            def get_action(obs)
            def choose_action(current_state)
            def get_max_action(current_state)
            def check_action_available(self, obs, action, func)
            def get_reward(obs, action)
            def get_end(obs)
            def check_done(obs, last_step)
'''

class Agent (DQNAgent):

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

    '''
        Initialize the agent
    '''
    def __init__(self, load=False):
        self.num_actions = len(self.possible_actions)
        self.num_states = 13

        # initialize neural network
        DQNAgent.__init__(self, 
                            num_actions=self.num_actions,
                            num_states=self.num_states,
                            episodes=EPISODES,
                            discount=0.99,
                            rep_mem_size=50_000,        # How many last steps to keep for model training
                            min_rep_mem_size=150,       # Minimum number of steps in a memory to start learning
                            learn_every=80,
                            update_time=50,             # When we'll copy weights from main network to target.
                            minibatch_size=64,
                            max_cases=1_000,            # Maximum number of cases until we start to learn
                            cases_to_delete=100,        # Cases to delete when we surpassed cases limit.
                            hidden_nodes=100,
                            num_hidden_layers=2,
                            load=load)
        
        if load:
            DQNAgent.loadModel(self, os.getcwd() + '\\deepQ\\models\\' + FILE_NAME + '.h5')

    '''
        Prepare basic parameters. This is called before start the episode.
    '''
    def prepare(self, obs, episode):
        DQNAgent.set_epsilon(self, episode=episode)

        self.enemy_totalHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        self.enemy_originalHP = self.enemy_totalHP
        self.enemy_onlyHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        self.ally_totalHP = self.__get_group_totalHP(obs, units.Protoss.Stalker)
        self.ally_originalHP = self.ally_totalHP
        self.last_dist = self.__get_dist(self.__get_meangroup_position(obs, units.Protoss.Stalker), self.__get_meangroup_position(obs, units.Protoss.Zealot))

        self.last_can_shoot = False
        self.dead = False

        return actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL]), 0
    
    '''
        Do step of the environment
    '''
    def step(self, env, func):
        obs = env.step(actions=[func])
        return obs, self.get_end(obs[0])

    '''
        Update values
    '''
    def update(self, obs, delta):
        self.last_can_shoot = self.current_can_shoot
        return
    
    '''
        Train agent
    '''
    def train(self, step, current_state, action, reward, new_state, done, epi = 0):
        # Every step we update replay memory and train main network
        DQNAgent.update_replay_memory(self, transition=(current_state, action, reward, new_state, done))
        DQNAgent.learn(self, step=step, ep=epi)
    
    '''
        Return agent state
        state:
        [UP, UP LEFT, LEFT, DOWN LEFT, DOWN, DOWN RIGHT, RIGHT, UP RIGHT, ------> enemy position
        UP WALL, LEFT WALL, DOWN WALL, RIGHT WALL,
        COOLDOWN]
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
        state = [0,0,0,0,0,0,0,0, 0, 0,0,0,0]

        # check dist
        dist = self.__get_dist([stalkerx, stalkery], [zealotx, zealoty])

        norm = 1 - ((dist - 4) / (55 - 5))
        norm = round(norm,1)
        # check angle
        if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
            state[0] = norm
        elif angleD >= 22.5 and angleD < 67.5:
            state[1] = norm
        elif angleD >= 67.5 and angleD < 112.5:
            state[2] = norm
        elif angleD >= 112.5 and angleD < 157.5:
            state[3] = norm
        elif angleD >= 157.5 and angleD < 202.5:
            state[4] = norm
        elif angleD >= 202.5 and angleD < 247.5:
            state[5] = norm
        elif angleD >= 247.5 and angleD < 292.5:
            state[6] = norm
        elif angleD >= 292.5 and angleD < 337.5:
            state[7] = norm

        # check limits
        if (stalkery - self._MOVE_VAL) < 3.5:
            state[8] = 1
        if (stalkerx - self._MOVE_VAL) < 3.5:
            state[9] = 1
        if (stalkery + self._MOVE_VAL) > 44.5:
            state[10] = 1
        if (stalkerx + self._MOVE_VAL) > 60.5:
            state[11] = 1

        self.current_can_shoot = True
        # check cooldown
        if self.__can_shoot(obs, units.Protoss.Stalker):
            state[12] = 1
            self.current_can_shoot = True
        else:
            self.current_can_shoot = False

        return state

    '''
        Return reward
    '''
    def get_reward(self, obs, action):
        reward = 0

        # reward for attacking
        actual_enemy_totalHP = self.__get_group_totalHP(obs, units.Protoss.Zealot)
        actual_ally_totalHP = self.__get_group_totalHP(obs, units.Protoss.Stalker)
        actual_enemy_onlyHP = self.__get_group_onlyHP(obs, units.Protoss.Zealot)

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
        Return if we must end this episode
    '''
    def get_end(self, obs):
        stalkers = self.__get_group(obs, units.Protoss.Stalker)
        self.dead = not stalkers
        return not stalkers

    '''
        Return
    '''
    def check_done(self, obs, last_step):
        stalkers = self.__get_group(obs, units.Protoss.Stalker)
        zealots = self.__get_group(obs, units.Protoss.Zealot)

        if last_step or not stalkers or not zealots:
            return True

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

    '''
        Return if current action is available in the environment
    '''
    def check_action_available(self, obs, action, func):
        if self.possible_actions[action] == self._ATTACK:
            # ATTACK ACTION
            if not (self._ATTACK_SCREEN in obs.observation.available_actions):
                func = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])
        else:
            if not (self._MOVE_SCREEN in obs.observation.available_actions):
                func = actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])
        return func

    '''
        (Private method)
        Return specified group
    '''
    def __get_group(self, obs, group_type):
        group = [unit for unit in obs.observation['feature_units'] 
                    if unit.unit_type == group_type]
        return group

    '''
        (Private method)
        Return zealot with lowest healt (health + shield)
    '''
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
        if len(group) < 0 or len(group) == 0:
                       
            unitx /= 1
            unity /= 1

        else:
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

   0/no_op                                              ()
   1/move_camera                                        (1/minimap [64, 64])
   2/select_point                                       (6/select_point_act [4]; 0/screen [84, 84])
   3/select_rect                                        (7/select_add [2]; 0/screen [84, 84]; 2/screen2 [84, 84])
   4/select_control_group                               (4/control_group_act [5]; 5/control_group_id [10])
   5/select_unit                                        (8/select_unit_act [4]; 9/select_unit_id [500])
   6/select_idle_worker                                 (10/select_worker [4])
   7/select_army                                        (7/select_add [2])
   8/select_warp_gates                                  (7/select_add [2])
   9/select_larva                                       ()
  10/unload                                             (12/unload_id [500])
  11/build_queue                                        (11/build_queue_id [10])
  12/Attack_screen                                      (3/queued [2]; 0/screen [84, 84])
  13/Attack_minimap                                     (3/queued [2]; 1/minimap [64, 64])
  14/Attack_Attack_screen                               (3/queued [2]; 0/screen [84, 84])
  15/Attack_Attack_minimap                              (3/queued [2]; 1/minimap [64, 64])
  16/Attack_AttackBuilding_screen                       (3/queued [2]; 0/screen [84, 84])
  17/Attack_AttackBuilding_minimap                      (3/queued [2]; 1/minimap [64, 64])
  18/Attack_Redirect_screen                             (3/queued [2]; 0/screen [84, 84])
  19/Scan_Move_screen                                   (3/queued [2]; 0/screen [84, 84])
  20/Scan_Move_minimap                                  (3/queued [2]; 1/minimap [64, 64])
  21/Behavior_BuildingAttackOff_quick                   (3/queued [2])
  22/Behavior_BuildingAttackOn_quick                    (3/queued [2])
  23/Behavior_CloakOff_quick                            (3/queued [2])
  24/Behavior_CloakOff_Banshee_quick                    (3/queued [2])
  25/Behavior_CloakOff_Ghost_quick                      (3/queued [2])
  26/Behavior_CloakOn_quick                             (3/queued [2])
  27/Behavior_CloakOn_Banshee_quick                     (3/queued [2])
  28/Behavior_CloakOn_Ghost_quick                       (3/queued [2])
  29/Behavior_GenerateCreepOff_quick                    (3/queued [2])
  30/Behavior_GenerateCreepOn_quick                     (3/queued [2])
  31/Behavior_HoldFireOff_quick                         (3/queued [2])
  32/Behavior_HoldFireOff_Ghost_quick                   (3/queued [2])
  33/Behavior_HoldFireOff_Lurker_quick                  (3/queued [2])
  34/Behavior_HoldFireOn_quick                          (3/queued [2])
  35/Behavior_HoldFireOn_Ghost_quick                    (3/queued [2])
  36/Behavior_HoldFireOn_Lurker_quick                   (3/queued [2])
  37/Behavior_PulsarBeamOff_quick                       (3/queued [2])
  38/Behavior_PulsarBeamOn_quick                        (3/queued [2])
  39/Build_Armory_screen                                (3/queued [2]; 0/screen [84, 84])
  40/Build_Assimilator_screen                           (3/queued [2]; 0/screen [84, 84])
  41/Build_BanelingNest_screen                          (3/queued [2]; 0/screen [84, 84])
  42/Build_Barracks_screen                              (3/queued [2]; 0/screen [84, 84])
  43/Build_Bunker_screen                                (3/queued [2]; 0/screen [84, 84])
  44/Build_CommandCenter_screen                         (3/queued [2]; 0/screen [84, 84])
  45/Build_CreepTumor_screen                            (3/queued [2]; 0/screen [84, 84])
  46/Build_CreepTumor_Queen_screen                      (3/queued [2]; 0/screen [84, 84])
  47/Build_CreepTumor_Tumor_screen                      (3/queued [2]; 0/screen [84, 84])
  48/Build_CyberneticsCore_screen                       (3/queued [2]; 0/screen [84, 84])
  49/Build_DarkShrine_screen                            (3/queued [2]; 0/screen [84, 84])
  50/Build_EngineeringBay_screen                        (3/queued [2]; 0/screen [84, 84])
  51/Build_EvolutionChamber_screen                      (3/queued [2]; 0/screen [84, 84])
  52/Build_Extractor_screen                             (3/queued [2]; 0/screen [84, 84])
  53/Build_Factory_screen                               (3/queued [2]; 0/screen [84, 84])
  54/Build_FleetBeacon_screen                           (3/queued [2]; 0/screen [84, 84])
  55/Build_Forge_screen                                 (3/queued [2]; 0/screen [84, 84])
  56/Build_FusionCore_screen                            (3/queued [2]; 0/screen [84, 84])
  57/Build_Gateway_screen                               (3/queued [2]; 0/screen [84, 84])
  58/Build_GhostAcademy_screen                          (3/queued [2]; 0/screen [84, 84])
  59/Build_Hatchery_screen                              (3/queued [2]; 0/screen [84, 84])
  60/Build_HydraliskDen_screen                          (3/queued [2]; 0/screen [84, 84])
  61/Build_InfestationPit_screen                        (3/queued [2]; 0/screen [84, 84])
  62/Build_Interceptors_quick                           (3/queued [2])
  63/Build_Interceptors_autocast                        ()
  64/Build_MissileTurret_screen                         (3/queued [2]; 0/screen [84, 84])
  65/Build_Nexus_screen                                 (3/queued [2]; 0/screen [84, 84])
  66/Build_Nuke_quick                                   (3/queued [2])
  67/Build_NydusNetwork_screen                          (3/queued [2]; 0/screen [84, 84])
  68/Build_NydusWorm_screen                             (3/queued [2]; 0/screen [84, 84])
  69/Build_PhotonCannon_screen                          (3/queued [2]; 0/screen [84, 84])
  70/Build_Pylon_screen                                 (3/queued [2]; 0/screen [84, 84])
  71/Build_Reactor_quick                                (3/queued [2])
  72/Build_Reactor_screen                               (3/queued [2]; 0/screen [84, 84])
  73/Build_Reactor_Barracks_quick                       (3/queued [2])
  74/Build_Reactor_Barracks_screen                      (3/queued [2]; 0/screen [84, 84])
  75/Build_Reactor_Factory_quick                        (3/queued [2])
  76/Build_Reactor_Factory_screen                       (3/queued [2]; 0/screen [84, 84])
  77/Build_Reactor_Starport_quick                       (3/queued [2])
  78/Build_Reactor_Starport_screen                      (3/queued [2]; 0/screen [84, 84])
  79/Build_Refinery_screen                              (3/queued [2]; 0/screen [84, 84])
  80/Build_RoachWarren_screen                           (3/queued [2]; 0/screen [84, 84])
  81/Build_RoboticsBay_screen                           (3/queued [2]; 0/screen [84, 84])
  82/Build_RoboticsFacility_screen                      (3/queued [2]; 0/screen [84, 84])
  83/Build_SensorTower_screen                           (3/queued [2]; 0/screen [84, 84])
  84/Build_SpawningPool_screen                          (3/queued [2]; 0/screen [84, 84])
  85/Build_SpineCrawler_screen                          (3/queued [2]; 0/screen [84, 84])
  86/Build_Spire_screen                                 (3/queued [2]; 0/screen [84, 84])
  87/Build_SporeCrawler_screen                          (3/queued [2]; 0/screen [84, 84])
  88/Build_Stargate_screen                              (3/queued [2]; 0/screen [84, 84])
  89/Build_Starport_screen                              (3/queued [2]; 0/screen [84, 84])
  90/Build_StasisTrap_screen                            (3/queued [2]; 0/screen [84, 84])
  91/Build_SupplyDepot_screen                           (3/queued [2]; 0/screen [84, 84])
  92/Build_TechLab_quick                                (3/queued [2])
  93/Build_TechLab_screen                               (3/queued [2]; 0/screen [84, 84])
  94/Build_TechLab_Barracks_quick                       (3/queued [2])
  95/Build_TechLab_Barracks_screen                      (3/queued [2]; 0/screen [84, 84])
  96/Build_TechLab_Factory_quick                        (3/queued [2])
  97/Build_TechLab_Factory_screen                       (3/queued [2]; 0/screen [84, 84])
  98/Build_TechLab_Starport_quick                       (3/queued [2])
  99/Build_TechLab_Starport_screen                      (3/queued [2]; 0/screen [84, 84])
 100/Build_TemplarArchive_screen                        (3/queued [2]; 0/screen [84, 84])
 101/Build_TwilightCouncil_screen                       (3/queued [2]; 0/screen [84, 84])
 102/Build_UltraliskCavern_screen                       (3/queued [2]; 0/screen [84, 84])
 103/BurrowDown_quick                                   (3/queued [2])
 104/BurrowDown_Baneling_quick                          (3/queued [2])
 105/BurrowDown_Drone_quick                             (3/queued [2])
 106/BurrowDown_Hydralisk_quick                         (3/queued [2])
 107/BurrowDown_Infestor_quick                          (3/queued [2])
 108/BurrowDown_InfestorTerran_quick                    (3/queued [2])
 109/BurrowDown_Lurker_quick                            (3/queued [2])
 110/BurrowDown_Queen_quick                             (3/queued [2])
 111/BurrowDown_Ravager_quick                           (3/queued [2])
 112/BurrowDown_Roach_quick                             (3/queued [2])
 113/BurrowDown_SwarmHost_quick                         (3/queued [2])
 114/BurrowDown_Ultralisk_quick                         (3/queued [2])
 115/BurrowDown_WidowMine_quick                         (3/queued [2])
 116/BurrowDown_Zergling_quick                          (3/queued [2])
 117/BurrowUp_quick                                     (3/queued [2])
 118/BurrowUp_autocast                                  ()
 119/BurrowUp_Baneling_quick                            (3/queued [2])
 120/BurrowUp_Baneling_autocast                         ()
 121/BurrowUp_Drone_quick                               (3/queued [2])
 122/BurrowUp_Hydralisk_quick                           (3/queued [2])
 123/BurrowUp_Hydralisk_autocast                        ()
 124/BurrowUp_Infestor_quick                            (3/queued [2])
 125/BurrowUp_InfestorTerran_quick                      (3/queued [2])
 126/BurrowUp_InfestorTerran_autocast                   ()
 127/BurrowUp_Lurker_quick                              (3/queued [2])
 128/BurrowUp_Queen_quick                               (3/queued [2])
 129/BurrowUp_Queen_autocast                            ()
 130/BurrowUp_Ravager_quick                             (3/queued [2])
 131/BurrowUp_Ravager_autocast                          ()
 132/BurrowUp_Roach_quick                               (3/queued [2])
 133/BurrowUp_Roach_autocast                            ()
 134/BurrowUp_SwarmHost_quick                           (3/queued [2])
 135/BurrowUp_Ultralisk_quick                           (3/queued [2])
 136/BurrowUp_Ultralisk_autocast                        ()
 137/BurrowUp_WidowMine_quick                           (3/queued [2])
 138/BurrowUp_Zergling_quick                            (3/queued [2])
 139/BurrowUp_Zergling_autocast                         ()
 140/Cancel_quick                                       (3/queued [2])
 141/Cancel_AdeptPhaseShift_quick                       (3/queued [2])
 142/Cancel_AdeptShadePhaseShift_quick                  (3/queued [2])
 143/Cancel_BarracksAddOn_quick                         (3/queued [2])
 144/Cancel_BuildInProgress_quick                       (3/queued [2])
 145/Cancel_CreepTumor_quick                            (3/queued [2])
 146/Cancel_FactoryAddOn_quick                          (3/queued [2])
 147/Cancel_GravitonBeam_quick                          (3/queued [2])
 148/Cancel_LockOn_quick                                (3/queued [2])
 149/Cancel_MorphBroodlord_quick                        (3/queued [2])
 150/Cancel_MorphGreaterSpire_quick                     (3/queued [2])
 151/Cancel_MorphHive_quick                             (3/queued [2])
 152/Cancel_MorphLair_quick                             (3/queued [2])
 153/Cancel_MorphLurker_quick                           (3/queued [2])
 154/Cancel_MorphLurkerDen_quick                        (3/queued [2])
 155/Cancel_MorphMothership_quick                       (3/queued [2])
 156/Cancel_MorphOrbital_quick                          (3/queued [2])
 157/Cancel_MorphOverlordTransport_quick                (3/queued [2])
 158/Cancel_MorphOverseer_quick                         (3/queued [2])
 159/Cancel_MorphPlanetaryFortress_quick                (3/queued [2])
 160/Cancel_MorphRavager_quick                          (3/queued [2])
 161/Cancel_MorphThorExplosiveMode_quick                (3/queued [2])
 162/Cancel_NeuralParasite_quick                        (3/queued [2])
 163/Cancel_Nuke_quick                                  (3/queued [2])
 164/Cancel_SpineCrawlerRoot_quick                      (3/queued [2])
 165/Cancel_SporeCrawlerRoot_quick                      (3/queued [2])
 166/Cancel_StarportAddOn_quick                         (3/queued [2])
 167/Cancel_StasisTrap_quick                            (3/queued [2])
 168/Cancel_Last_quick                                  (3/queued [2])
 169/Cancel_HangarQueue5_quick                          (3/queued [2])
 170/Cancel_Queue1_quick                                (3/queued [2])
 171/Cancel_Queue5_quick                                (3/queued [2])
 172/Cancel_QueueAddOn_quick                            (3/queued [2])
 173/Cancel_QueueCancelToSelection_quick                (3/queued [2])
 174/Cancel_QueuePassive_quick                          (3/queued [2])
 175/Cancel_QueuePassiveCancelToSelection_quick         (3/queued [2])
 176/Effect_Abduct_screen                               (3/queued [2]; 0/screen [84, 84])
 177/Effect_AdeptPhaseShift_screen                      (3/queued [2]; 0/screen [84, 84])
 178/Effect_AutoTurret_screen                           (3/queued [2]; 0/screen [84, 84])
 179/Effect_BlindingCloud_screen                        (3/queued [2]; 0/screen [84, 84])
 180/Effect_Blink_screen                                (3/queued [2]; 0/screen [84, 84])
 181/Effect_Blink_Stalker_screen                        (3/queued [2]; 0/screen [84, 84])
 182/Effect_ShadowStride_screen                         (3/queued [2]; 0/screen [84, 84])
 183/Effect_CalldownMULE_screen                         (3/queued [2]; 0/screen [84, 84])
 184/Effect_CausticSpray_screen                         (3/queued [2]; 0/screen [84, 84])
 185/Effect_Charge_screen                               (3/queued [2]; 0/screen [84, 84])
 186/Effect_Charge_autocast                             ()
 187/Effect_ChronoBoost_screen                          (3/queued [2]; 0/screen [84, 84])
 188/Effect_Contaminate_screen                          (3/queued [2]; 0/screen [84, 84])
 189/Effect_CorrosiveBile_screen                        (3/queued [2]; 0/screen [84, 84])
 190/Effect_EMP_screen                                  (3/queued [2]; 0/screen [84, 84])
 191/Effect_Explode_quick                               (3/queued [2])
 192/Effect_Feedback_screen                             (3/queued [2]; 0/screen [84, 84])
 193/Effect_ForceField_screen                           (3/queued [2]; 0/screen [84, 84])
 194/Effect_FungalGrowth_screen                         (3/queued [2]; 0/screen [84, 84])
 195/Effect_GhostSnipe_screen                           (3/queued [2]; 0/screen [84, 84])
 196/Effect_GravitonBeam_screen                         (3/queued [2]; 0/screen [84, 84])
 197/Effect_GuardianShield_quick                        (3/queued [2])
 198/Effect_Heal_screen                                 (3/queued [2]; 0/screen [84, 84])
 199/Effect_Heal_autocast                               ()
 200/Effect_HunterSeekerMissile_screen                  (3/queued [2]; 0/screen [84, 84])
 201/Effect_ImmortalBarrier_quick                       (3/queued [2])
 202/Effect_ImmortalBarrier_autocast                    ()
 203/Effect_InfestedTerrans_screen                      (3/queued [2]; 0/screen [84, 84])
 204/Effect_InjectLarva_screen                          (3/queued [2]; 0/screen [84, 84])
 205/Effect_KD8Charge_screen                            (3/queued [2]; 0/screen [84, 84])
 206/Effect_LockOn_screen                               (3/queued [2]; 0/screen [84, 84])
 207/Effect_LocustSwoop_screen                          (3/queued [2]; 0/screen [84, 84])
 208/Effect_MassRecall_screen                           (3/queued [2]; 0/screen [84, 84])
 209/Effect_MassRecall_Mothership_screen                (3/queued [2]; 0/screen [84, 84])
 210/Effect_MassRecall_MothershipCore_screen            (3/queued [2]; 0/screen [84, 84])
 211/Effect_MedivacIgniteAfterburners_quick             (3/queued [2])
 212/Effect_NeuralParasite_screen                       (3/queued [2]; 0/screen [84, 84])
 213/Effect_NukeCalldown_screen                         (3/queued [2]; 0/screen [84, 84])
 214/Effect_OracleRevelation_screen                     (3/queued [2]; 0/screen [84, 84])
 215/Effect_ParasiticBomb_screen                        (3/queued [2]; 0/screen [84, 84])
 216/Effect_PhotonOvercharge_screen                     (3/queued [2]; 0/screen [84, 84])
 217/Effect_PointDefenseDrone_screen                    (3/queued [2]; 0/screen [84, 84])
 218/Effect_PsiStorm_screen                             (3/queued [2]; 0/screen [84, 84])
 219/Effect_PurificationNova_screen                     (3/queued [2]; 0/screen [84, 84])
 220/Effect_Repair_screen                               (3/queued [2]; 0/screen [84, 84])
 221/Effect_Repair_autocast                             ()
 222/Effect_Repair_Mule_screen                          (3/queued [2]; 0/screen [84, 84])
 223/Effect_Repair_Mule_autocast                        ()
 224/Effect_Repair_SCV_screen                           (3/queued [2]; 0/screen [84, 84])
 225/Effect_Repair_SCV_autocast                         ()
 226/Effect_Salvage_quick                               (3/queued [2])
 227/Effect_Scan_screen                                 (3/queued [2]; 0/screen [84, 84])
 228/Effect_SpawnChangeling_quick                       (3/queued [2])
 229/Effect_SpawnLocusts_screen                         (3/queued [2]; 0/screen [84, 84])
 230/Effect_Spray_screen                                (3/queued [2]; 0/screen [84, 84])
 231/Effect_Spray_Protoss_screen                        (3/queued [2]; 0/screen [84, 84])
 232/Effect_Spray_Terran_screen                         (3/queued [2]; 0/screen [84, 84])
 233/Effect_Spray_Zerg_screen                           (3/queued [2]; 0/screen [84, 84])
 234/Effect_Stim_quick                                  (3/queued [2])
 235/Effect_Stim_Marauder_quick                         (3/queued [2])
 236/Effect_Stim_Marauder_Redirect_quick                (3/queued [2])
 237/Effect_Stim_Marine_quick                           (3/queued [2])
 238/Effect_Stim_Marine_Redirect_quick                  (3/queued [2])
 239/Effect_SupplyDrop_screen                           (3/queued [2]; 0/screen [84, 84])
 240/Effect_TacticalJump_screen                         (3/queued [2]; 0/screen [84, 84])
 241/Effect_TimeWarp_screen                             (3/queued [2]; 0/screen [84, 84])
 242/Effect_Transfusion_screen                          (3/queued [2]; 0/screen [84, 84])
 243/Effect_ViperConsume_screen                         (3/queued [2]; 0/screen [84, 84])
 244/Effect_VoidRayPrismaticAlignment_quick             (3/queued [2])
 245/Effect_WidowMineAttack_screen                      (3/queued [2]; 0/screen [84, 84])
 246/Effect_WidowMineAttack_autocast                    ()
 247/Effect_YamatoGun_screen                            (3/queued [2]; 0/screen [84, 84])
 248/Hallucination_Adept_quick                          (3/queued [2])
 249/Hallucination_Archon_quick                         (3/queued [2])
 250/Hallucination_Colossus_quick                       (3/queued [2])
 251/Hallucination_Disruptor_quick                      (3/queued [2])
 252/Hallucination_HighTemplar_quick                    (3/queued [2])
 253/Hallucination_Immortal_quick                       (3/queued [2])
 254/Hallucination_Oracle_quick                         (3/queued [2])
 255/Hallucination_Phoenix_quick                        (3/queued [2])
 256/Hallucination_Probe_quick                          (3/queued [2])
 257/Hallucination_Stalker_quick                        (3/queued [2])
 258/Hallucination_VoidRay_quick                        (3/queued [2])
 259/Hallucination_WarpPrism_quick                      (3/queued [2])
 260/Hallucination_Zealot_quick                         (3/queued [2])
 261/Halt_quick                                         (3/queued [2])
 262/Halt_Building_quick                                (3/queued [2])
 263/Halt_TerranBuild_quick                             (3/queued [2])
 264/Harvest_Gather_screen                              (3/queued [2]; 0/screen [84, 84])
 265/Harvest_Gather_Drone_screen                        (3/queued [2]; 0/screen [84, 84])
 266/Harvest_Gather_Mule_screen                         (3/queued [2]; 0/screen [84, 84])
 267/Harvest_Gather_Probe_screen                        (3/queued [2]; 0/screen [84, 84])
 268/Harvest_Gather_SCV_screen                          (3/queued [2]; 0/screen [84, 84])
 269/Harvest_Return_quick                               (3/queued [2])
 270/Harvest_Return_Drone_quick                         (3/queued [2])
 271/Harvest_Return_Mule_quick                          (3/queued [2])
 272/Harvest_Return_Probe_quick                         (3/queued [2])
 273/Harvest_Return_SCV_quick                           (3/queued [2])
 274/HoldPosition_quick                                 (3/queued [2])
 275/Land_screen                                        (3/queued [2]; 0/screen [84, 84])
 276/Land_Barracks_screen                               (3/queued [2]; 0/screen [84, 84])
 277/Land_CommandCenter_screen                          (3/queued [2]; 0/screen [84, 84])
 278/Land_Factory_screen                                (3/queued [2]; 0/screen [84, 84])
 279/Land_OrbitalCommand_screen                         (3/queued [2]; 0/screen [84, 84])
 280/Land_Starport_screen                               (3/queued [2]; 0/screen [84, 84])
 281/Lift_quick                                         (3/queued [2])
 282/Lift_Barracks_quick                                (3/queued [2])
 283/Lift_CommandCenter_quick                           (3/queued [2])
 284/Lift_Factory_quick                                 (3/queued [2])
 285/Lift_OrbitalCommand_quick                          (3/queued [2])
 286/Lift_Starport_quick                                (3/queued [2])
 287/Load_screen                                        (3/queued [2]; 0/screen [84, 84])
 288/Load_Bunker_screen                                 (3/queued [2]; 0/screen [84, 84])
 289/Load_Medivac_screen                                (3/queued [2]; 0/screen [84, 84])
 290/Load_NydusNetwork_screen                           (3/queued [2]; 0/screen [84, 84])
 291/Load_NydusWorm_screen                              (3/queued [2]; 0/screen [84, 84])
 292/Load_Overlord_screen                               (3/queued [2]; 0/screen [84, 84])
 293/Load_WarpPrism_screen                              (3/queued [2]; 0/screen [84, 84])
 294/LoadAll_quick                                      (3/queued [2])
 295/LoadAll_CommandCenter_quick                        (3/queued [2])
 296/Morph_Archon_quick                                 (3/queued [2])
 297/Morph_BroodLord_quick                              (3/queued [2])
 298/Morph_Gateway_quick                                (3/queued [2])
 299/Morph_GreaterSpire_quick                           (3/queued [2])
 300/Morph_Hellbat_quick                                (3/queued [2])
 301/Morph_Hellion_quick                                (3/queued [2])
 302/Morph_Hive_quick                                   (3/queued [2])
 303/Morph_Lair_quick                                   (3/queued [2])
 304/Morph_LiberatorAAMode_quick                        (3/queued [2])
 305/Morph_LiberatorAGMode_screen                       (3/queued [2]; 0/screen [84, 84])
 306/Morph_Lurker_quick                                 (3/queued [2])
 307/Morph_LurkerDen_quick                              (3/queued [2])
 308/Morph_Mothership_quick                             (3/queued [2])
 309/Morph_OrbitalCommand_quick                         (3/queued [2])
 310/Morph_OverlordTransport_quick                      (3/queued [2])
 311/Morph_Overseer_quick                               (3/queued [2])
 312/Morph_PlanetaryFortress_quick                      (3/queued [2])
 313/Morph_Ravager_quick                                (3/queued [2])
 314/Morph_Root_screen                                  (3/queued [2]; 0/screen [84, 84])
 315/Morph_SpineCrawlerRoot_screen                      (3/queued [2]; 0/screen [84, 84])
 316/Morph_SporeCrawlerRoot_screen                      (3/queued [2]; 0/screen [84, 84])
 317/Morph_SiegeMode_quick                              (3/queued [2])
 318/Morph_SupplyDepot_Lower_quick                      (3/queued [2])
 319/Morph_SupplyDepot_Raise_quick                      (3/queued [2])
 320/Morph_ThorExplosiveMode_quick                      (3/queued [2])
 321/Morph_ThorHighImpactMode_quick                     (3/queued [2])
 322/Morph_Unsiege_quick                                (3/queued [2])
 323/Morph_Uproot_quick                                 (3/queued [2])
 324/Morph_SpineCrawlerUproot_quick                     (3/queued [2])
 325/Morph_SporeCrawlerUproot_quick                     (3/queued [2])
 326/Morph_VikingAssaultMode_quick                      (3/queued [2])
 327/Morph_VikingFighterMode_quick                      (3/queued [2])
 328/Morph_WarpGate_quick                               (3/queued [2])
 329/Morph_WarpPrismPhasingMode_quick                   (3/queued [2])
 330/Morph_WarpPrismTransportMode_quick                 (3/queued [2])
 331/Move_screen                                        (3/queued [2]; 0/screen [84, 84])
 332/Move_minimap                                       (3/queued [2]; 1/minimap [64, 64])
 333/Patrol_screen                                      (3/queued [2]; 0/screen [84, 84])
 334/Patrol_minimap                                     (3/queued [2]; 1/minimap [64, 64])
 335/Rally_Units_screen                                 (3/queued [2]; 0/screen [84, 84])
 336/Rally_Units_minimap                                (3/queued [2]; 1/minimap [64, 64])
 337/Rally_Building_screen                              (3/queued [2]; 0/screen [84, 84])
 338/Rally_Building_minimap                             (3/queued [2]; 1/minimap [64, 64])
 339/Rally_Hatchery_Units_screen                        (3/queued [2]; 0/screen [84, 84])
 340/Rally_Hatchery_Units_minimap                       (3/queued [2]; 1/minimap [64, 64])
 341/Rally_Morphing_Unit_screen                         (3/queued [2]; 0/screen [84, 84])
 342/Rally_Morphing_Unit_minimap                        (3/queued [2]; 1/minimap [64, 64])
 343/Rally_Workers_screen                               (3/queued [2]; 0/screen [84, 84])
 344/Rally_Workers_minimap                              (3/queued [2]; 1/minimap [64, 64])
 345/Rally_CommandCenter_screen                         (3/queued [2]; 0/screen [84, 84])
 346/Rally_CommandCenter_minimap                        (3/queued [2]; 1/minimap [64, 64])
 347/Rally_Hatchery_Workers_screen                      (3/queued [2]; 0/screen [84, 84])
 348/Rally_Hatchery_Workers_minimap                     (3/queued [2]; 1/minimap [64, 64])
 349/Rally_Nexus_screen                                 (3/queued [2]; 0/screen [84, 84])
 350/Rally_Nexus_minimap                                (3/queued [2]; 1/minimap [64, 64])
 351/Research_AdeptResonatingGlaives_quick              (3/queued [2])
 352/Research_AdvancedBallistics_quick                  (3/queued [2])
 353/Research_BansheeCloakingField_quick                (3/queued [2])
 354/Research_BansheeHyperflightRotors_quick            (3/queued [2])
 355/Research_BattlecruiserWeaponRefit_quick            (3/queued [2])
 356/Research_Blink_quick                               (3/queued [2])
 357/Research_Burrow_quick                              (3/queued [2])
 358/Research_CentrifugalHooks_quick                    (3/queued [2])
 359/Research_Charge_quick                              (3/queued [2])
 360/Research_ChitinousPlating_quick                    (3/queued [2])
 361/Research_CombatShield_quick                        (3/queued [2])
 362/Research_ConcussiveShells_quick                    (3/queued [2])
 363/Research_DrillingClaws_quick                       (3/queued [2])
 364/Research_ExtendedThermalLance_quick                (3/queued [2])
 365/Research_GlialRegeneration_quick                   (3/queued [2])
 366/Research_GraviticBooster_quick                     (3/queued [2])
 367/Research_GraviticDrive_quick                       (3/queued [2])
 368/Research_GroovedSpines_quick                       (3/queued [2])
 369/Research_HiSecAutoTracking_quick                   (3/queued [2])
 370/Research_HighCapacityFuelTanks_quick               (3/queued [2])
 371/Research_InfernalPreigniter_quick                  (3/queued [2])
 372/Research_InterceptorGravitonCatapult_quick         (3/queued [2])
 373/Research_SmartServos_quick                         (3/queued [2])
 374/Research_MuscularAugments_quick                    (3/queued [2])
 375/Research_NeosteelFrame_quick                       (3/queued [2])
 376/Research_NeuralParasite_quick                      (3/queued [2])
 377/Research_PathogenGlands_quick                      (3/queued [2])
 378/Research_PersonalCloaking_quick                    (3/queued [2])
 379/Research_PhoenixAnionPulseCrystals_quick           (3/queued [2])
 380/Research_PneumatizedCarapace_quick                 (3/queued [2])
 381/Research_ProtossAirArmor_quick                     (3/queued [2])
 382/Research_ProtossAirArmorLevel1_quick               (3/queued [2])
 383/Research_ProtossAirArmorLevel2_quick               (3/queued [2])
 384/Research_ProtossAirArmorLevel3_quick               (3/queued [2])
 385/Research_ProtossAirWeapons_quick                   (3/queued [2])
 386/Research_ProtossAirWeaponsLevel1_quick             (3/queued [2])
 387/Research_ProtossAirWeaponsLevel2_quick             (3/queued [2])
 388/Research_ProtossAirWeaponsLevel3_quick             (3/queued [2])
 389/Research_ProtossGroundArmor_quick                  (3/queued [2])
 390/Research_ProtossGroundArmorLevel1_quick            (3/queued [2])
 391/Research_ProtossGroundArmorLevel2_quick            (3/queued [2])
 392/Research_ProtossGroundArmorLevel3_quick            (3/queued [2])
 393/Research_ProtossGroundWeapons_quick                (3/queued [2])
 394/Research_ProtossGroundWeaponsLevel1_quick          (3/queued [2])
 395/Research_ProtossGroundWeaponsLevel2_quick          (3/queued [2])
 396/Research_ProtossGroundWeaponsLevel3_quick          (3/queued [2])
 397/Research_ProtossShields_quick                      (3/queued [2])
 398/Research_ProtossShieldsLevel1_quick                (3/queued [2])
 399/Research_ProtossShieldsLevel2_quick                (3/queued [2])
 400/Research_ProtossShieldsLevel3_quick                (3/queued [2])
 401/Research_PsiStorm_quick                            (3/queued [2])
 402/Research_RavenCorvidReactor_quick                  (3/queued [2])
 403/Research_RavenRecalibratedExplosives_quick         (3/queued [2])
 404/Research_ShadowStrike_quick                        (3/queued [2])
 405/Research_Stimpack_quick                            (3/queued [2])
 406/Research_TerranInfantryArmor_quick                 (3/queued [2])
 407/Research_TerranInfantryArmorLevel1_quick           (3/queued [2])
 408/Research_TerranInfantryArmorLevel2_quick           (3/queued [2])
 409/Research_TerranInfantryArmorLevel3_quick           (3/queued [2])
 410/Research_TerranInfantryWeapons_quick               (3/queued [2])
 411/Research_TerranInfantryWeaponsLevel1_quick         (3/queued [2])
 412/Research_TerranInfantryWeaponsLevel2_quick         (3/queued [2])
 413/Research_TerranInfantryWeaponsLevel3_quick         (3/queued [2])
 414/Research_TerranShipWeapons_quick                   (3/queued [2])
 415/Research_TerranShipWeaponsLevel1_quick             (3/queued [2])
 416/Research_TerranShipWeaponsLevel2_quick             (3/queued [2])
 417/Research_TerranShipWeaponsLevel3_quick             (3/queued [2])
 418/Research_TerranStructureArmorUpgrade_quick         (3/queued [2])
 419/Research_TerranVehicleAndShipPlating_quick         (3/queued [2])
 420/Research_TerranVehicleAndShipPlatingLevel1_quick   (3/queued [2])
 421/Research_TerranVehicleAndShipPlatingLevel2_quick   (3/queued [2])
 422/Research_TerranVehicleAndShipPlatingLevel3_quick   (3/queued [2])
 423/Research_TerranVehicleWeapons_quick                (3/queued [2])
 424/Research_TerranVehicleWeaponsLevel1_quick          (3/queued [2])
 425/Research_TerranVehicleWeaponsLevel2_quick          (3/queued [2])
 426/Research_TerranVehicleWeaponsLevel3_quick          (3/queued [2])
 427/Research_TunnelingClaws_quick                      (3/queued [2])
 428/Research_WarpGate_quick                            (3/queued [2])
 429/Research_ZergFlyerArmor_quick                      (3/queued [2])
 430/Research_ZergFlyerArmorLevel1_quick                (3/queued [2])
 431/Research_ZergFlyerArmorLevel2_quick                (3/queued [2])
 432/Research_ZergFlyerArmorLevel3_quick                (3/queued [2])
 433/Research_ZergFlyerAttack_quick                     (3/queued [2])
 434/Research_ZergFlyerAttackLevel1_quick               (3/queued [2])
 435/Research_ZergFlyerAttackLevel2_quick               (3/queued [2])
 436/Research_ZergFlyerAttackLevel3_quick               (3/queued [2])
 437/Research_ZergGroundArmor_quick                     (3/queued [2])
 438/Research_ZergGroundArmorLevel1_quick               (3/queued [2])
 439/Research_ZergGroundArmorLevel2_quick               (3/queued [2])
 440/Research_ZergGroundArmorLevel3_quick               (3/queued [2])
 441/Research_ZergMeleeWeapons_quick                    (3/queued [2])
 442/Research_ZergMeleeWeaponsLevel1_quick              (3/queued [2])
 443/Research_ZergMeleeWeaponsLevel2_quick              (3/queued [2])
 444/Research_ZergMeleeWeaponsLevel3_quick              (3/queued [2])
 445/Research_ZergMissileWeapons_quick                  (3/queued [2])
 446/Research_ZergMissileWeaponsLevel1_quick            (3/queued [2])
 447/Research_ZergMissileWeaponsLevel2_quick            (3/queued [2])
 448/Research_ZergMissileWeaponsLevel3_quick            (3/queued [2])
 449/Research_ZerglingAdrenalGlands_quick               (3/queued [2])
 450/Research_ZerglingMetabolicBoost_quick              (3/queued [2])
 451/Smart_screen                                       (3/queued [2]; 0/screen [84, 84])
 452/Smart_minimap                                      (3/queued [2]; 1/minimap [64, 64])
 453/Stop_quick                                         (3/queued [2])
 454/Stop_Building_quick                                (3/queued [2])
 455/Stop_Redirect_quick                                (3/queued [2])
 456/Stop_Stop_quick                                    (3/queued [2])
 457/Train_Adept_quick                                  (3/queued [2])
 458/Train_Baneling_quick                               (3/queued [2])
 459/Train_Banshee_quick                                (3/queued [2])
 460/Train_Battlecruiser_quick                          (3/queued [2])
 461/Train_Carrier_quick                                (3/queued [2])
 462/Train_Colossus_quick                               (3/queued [2])
 463/Train_Corruptor_quick                              (3/queued [2])
 464/Train_Cyclone_quick                                (3/queued [2])
 465/Train_DarkTemplar_quick                            (3/queued [2])
 466/Train_Disruptor_quick                              (3/queued [2])
 467/Train_Drone_quick                                  (3/queued [2])
 468/Train_Ghost_quick                                  (3/queued [2])
 469/Train_Hellbat_quick                                (3/queued [2])
 470/Train_Hellion_quick                                (3/queued [2])
 471/Train_HighTemplar_quick                            (3/queued [2])
 472/Train_Hydralisk_quick                              (3/queued [2])
 473/Train_Immortal_quick                               (3/queued [2])
 474/Train_Infestor_quick                               (3/queued [2])
 475/Train_Liberator_quick                              (3/queued [2])
 476/Train_Marauder_quick                               (3/queued [2])
 477/Train_Marine_quick                                 (3/queued [2])
 478/Train_Medivac_quick                                (3/queued [2])
 479/Train_MothershipCore_quick                         (3/queued [2])
 480/Train_Mutalisk_quick                               (3/queued [2])
 481/Train_Observer_quick                               (3/queued [2])
 482/Train_Oracle_quick                                 (3/queued [2])
 483/Train_Overlord_quick                               (3/queued [2])
 484/Train_Phoenix_quick                                (3/queued [2])
 485/Train_Probe_quick                                  (3/queued [2])
 486/Train_Queen_quick                                  (3/queued [2])
 487/Train_Raven_quick                                  (3/queued [2])
 488/Train_Reaper_quick                                 (3/queued [2])
 489/Train_Roach_quick                                  (3/queued [2])
 490/Train_SCV_quick                                    (3/queued [2])
 491/Train_Sentry_quick                                 (3/queued [2])
 492/Train_SiegeTank_quick                              (3/queued [2])
 493/Train_Stalker_quick                                (3/queued [2])
 494/Train_SwarmHost_quick                              (3/queued [2])
 495/Train_Tempest_quick                                (3/queued [2])
 496/Train_Thor_quick                                   (3/queued [2])
 497/Train_Ultralisk_quick                              (3/queued [2])
 498/Train_VikingFighter_quick                          (3/queued [2])
 499/Train_Viper_quick                                  (3/queued [2])
 500/Train_VoidRay_quick                                (3/queued [2])
 501/Train_WarpPrism_quick                              (3/queued [2])
 502/Train_WidowMine_quick                              (3/queued [2])
 503/Train_Zealot_quick                                 (3/queued [2])
 504/Train_Zergling_quick                               (3/queued [2])
 505/TrainWarp_Adept_screen                             (3/queued [2]; 0/screen [84, 84])
 506/TrainWarp_DarkTemplar_screen                       (3/queued [2]; 0/screen [84, 84])
 507/TrainWarp_HighTemplar_screen                       (3/queued [2]; 0/screen [84, 84])
 508/TrainWarp_Sentry_screen                            (3/queued [2]; 0/screen [84, 84])
 509/TrainWarp_Stalker_screen                           (3/queued [2]; 0/screen [84, 84])
 510/TrainWarp_Zealot_screen                            (3/queued [2]; 0/screen [84, 84])
 511/UnloadAll_quick                                    (3/queued [2])
 512/UnloadAll_Bunker_quick                             (3/queued [2])
 513/UnloadAll_CommandCenter_quick                      (3/queued [2])
 514/UnloadAll_NydusNetwork_quick                       (3/queued [2])
 515/UnloadAll_NydusWorm_quick                          (3/queued [2])
 516/UnloadAllAt_screen                                 (3/queued [2]; 0/screen [84, 84])
 517/UnloadAllAt_minimap                                (3/queued [2]; 1/minimap [64, 64])
 518/UnloadAllAt_Medivac_screen                         (3/queued [2]; 0/screen [84, 84])
 519/UnloadAllAt_Medivac_minimap                        (3/queued [2]; 1/minimap [64, 64])
 520/UnloadAllAt_Overlord_screen                        (3/queued [2]; 0/screen [84, 84])
 521/UnloadAllAt_Overlord_minimap                       (3/queued [2]; 1/minimap [64, 64])
 522/UnloadAllAt_WarpPrism_screen                       (3/queued [2]; 0/screen [84, 84])
 523/UnloadAllAt_WarpPrism_minimap                      (3/queued [2]; 1/minimap [64, 64])
 524/Build_LurkerDen_screen                             (3/queued [2]; 0/screen [84, 84])
 525/Build_ShieldBattery_screen                         (3/queued [2]; 0/screen [84, 84])
 526/Effect_AntiArmorMissile_screen                     (3/queued [2]; 0/screen [84, 84])
 527/Effect_ChronoBoostEnergyCost_screen                (3/queued [2]; 0/screen [84, 84])
 528/Effect_InterferenceMatrix_screen                   (3/queued [2]; 0/screen [84, 84])
 529/Effect_MassRecall_Nexus_screen                     (3/queued [2]; 0/screen [84, 84])
 530/Effect_Repair_RepairDrone_screen                   (3/queued [2]; 0/screen [84, 84])
 531/Effect_Repair_RepairDrone_autocast                 ()
 532/Effect_RepairDrone_screen                          (3/queued [2]; 0/screen [84, 84])
 533/Effect_Restore_screen                              (3/queued [2]; 0/screen [84, 84])
 534/Effect_Restore_autocast                            ()
 535/Morph_ObserverMode_quick                           (3/queued [2])
 536/Morph_OverseerMode_quick                           (3/queued [2])
 537/Morph_OversightMode_quick                          (3/queued [2])
 538/Morph_SurveillanceMode_quick                       (3/queued [2])
 539/Research_AdaptiveTalons_quick                      (3/queued [2])
 540/Research_CycloneRapidFireLaunchers_quick           (3/queued [2])
 541/Train_Mothership_quick                             (3/queued [2])
 542/Effect_Scan_minimap                                (3/queued [2]; 1/minimap [64, 64])
 543/Effect_Blink_minimap                               (3/queued [2]; 1/minimap [64, 64])
 544/Effect_Blink_Stalker_minimap                       (3/queued [2]; 1/minimap [64, 64])
 545/Effect_ShadowStride_minimap                        (3/queued [2]; 1/minimap [64, 64])
 546/Cancel_VoidRayPrismaticAlignment_quick             (3/queued [2])
 547/Effect_AdeptPhaseShift_minimap                     (3/queued [2]; 1/minimap [64, 64])
 548/Effect_MassRecall_StrategicRecall_screen           (3/queued [2]; 0/screen [84, 84])
 549/Effect_Spray_minimap                               (3/queued [2]; 1/minimap [64, 64])
 550/Effect_Spray_Protoss_minimap                       (3/queued [2]; 1/minimap [64, 64])
 551/Effect_Spray_Terran_minimap                        (3/queued [2]; 1/minimap [64, 64])
 552/Effect_Spray_Zerg_minimap                          (3/queued [2]; 1/minimap [64, 64])
 553/Effect_TacticalJump_minimap                        (3/queued [2]; 1/minimap [64, 64])
 554/Morph_LiberatorAGMode_minimap                      (3/queued [2]; 1/minimap [64, 64])
 555/Attack_Battlecruiser_screen                        (3/queued [2]; 0/screen [84, 84])
 556/Attack_Battlecruiser_minimap                       (3/queued [2]; 1/minimap [64, 64])
 557/Effect_LockOn_autocast                             ()
 558/HoldPosition_Battlecruiser_quick                   (3/queued [2])
 559/HoldPosition_Hold_quick                            (3/queued [2])
 560/Morph_WarpGate_autocast                            ()
 561/Move_Battlecruiser_screen                          (3/queued [2]; 0/screen [84, 84])
 562/Move_Battlecruiser_minimap                         (3/queued [2]; 1/minimap [64, 64])
 563/Move_Move_screen                                   (3/queued [2]; 0/screen [84, 84])
 564/Move_Move_minimap                                  (3/queued [2]; 1/minimap [64, 64])
 565/Patrol_Battlecruiser_screen                        (3/queued [2]; 0/screen [84, 84])
 566/Patrol_Battlecruiser_minimap                       (3/queued [2]; 1/minimap [64, 64])
 567/Patrol_Patrol_screen                               (3/queued [2]; 0/screen [84, 84])
 568/Patrol_Patrol_minimap                              (3/queued [2]; 1/minimap [64, 64])
 569/Research_AnabolicSynthesis_quick                   (3/queued [2])
 570/Research_CycloneLockOnDamage_quick                 (3/queued [2])
 571/Stop_Battlecruiser_quick                           (3/queued [2])
 572/Research_EnhancedShockwaves_quick                  (3/queued [2])