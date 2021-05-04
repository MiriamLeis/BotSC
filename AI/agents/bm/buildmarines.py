import numpy as np
import math
import random 
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

from agents.abstract_base import AbstractBase

class BuildMarines(AbstractBase): 
    _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
    _PLAYER_SELF = 1
    _PLAYER_NEUTRAL = 3

    _NO_OP = actions.FUNCTIONS.no_op.id
    _MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
    _SELECT_ARMY = actions.FUNCTIONS.select_army.id
    _SELECT_POINT = actions.FUNCTIONS.select_point.id
    _SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
    _HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
    _TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
    _TRAIN_WORKER = actions.FUNCTIONS.Train_SCV_quick.id
    _RALLY_WORKERS_SCREEN = actions.FUNCTIONS.Rally_Workers_screen.id
    _RALLY_UNITS_SCREEN = actions.FUNCTIONS.Rally_Units_screen.id 

    _BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
    _BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id

    _TERRAN_COMMANDCENTER = units.Terran.CommandCenter
    _TERRAN_SCV = units.Terran.SCV
    _TERRAN_MARINE = units.Terran.Marine
    _TERRAN_BARRACKS = units.Terran.Barracks
    _TERRAN_SUPLY_DEPOT = units.Terran.SupplyDepot
    _MINERAL_FIELD = units.Neutral.MineralField

    _IDLE_WORKER_HARVEST = 0
    _IDLE_WORKER_BUILD_HOUSE = 1
    _IDLE_WORKER_BUILD_BARRACKS = 2
    _RANDOM_WORKER_BUILD_HOUSE = 3
    _RANDOM_WORKER_BUILD_BARRACKS = 4
    _CREATE_WORKER = 5
    _CREATE_MARINES = 6
    _DO_NOTHING = 7

    _SELECT_ALL = [0]
    _NOT_QUEUED = [0]
    _QUEUED = [1]

    _MOVE_VAL = 3.5

    possible_actions = [
        _IDLE_WORKER_HARVEST,
        _IDLE_WORKER_BUILD_HOUSE,
        _IDLE_WORKER_BUILD_BARRACKS,
        _RANDOM_WORKER_BUILD_HOUSE,
        _RANDOM_WORKER_BUILD_BARRACKS,
        _CREATE_WORKER,
        _CREATE_MARINES,
        _DO_NOTHING
    ]

    def __init__(self):
        super().__init__()

    def get_args(self):
        super().get_args()

        return ['BuildMarines']
    
    '''
        Return basic information.
    '''
    def get_info(self):
        return {
            'actions' : self.possible_actions, 
             'num_states' : 31,
             'discount' : 0.99,
             'replay_mem_size' : 50_000,
             'learn_every' : 150,
             'min_replay_mem_size' : 1024,
             'minibatch_size' : 264,
             'update_time' : 10,
             'max_cases' : 2048,
             'cases_to_delete' : 128,
             'hidden_nodes' : 100,
             'hidden_layer' : 3}

    '''
        Prepare basic parameters.
    '''
    def prepare(self, env):
        super().prepare(env=env)

        self.maxHouses = 20
        self.maxBarracks = 8
        self.action = actions.FunctionCall(self._NO_OP, [])
        self.actualMarines = 0

        return 0

    '''
        Update basic values and train
    '''
    def update(self, env, deltaTime):
        super().update(env=env, deltaTime=deltaTime)

    '''
        Do step of the environment
    '''
    def step(self, env, environment):
        for func in self.action:
            finalAct = self.__check_action_available(env=env, action=func)
            env = environment.step(actions=[finalAct])
        self.action = [actions.FunctionCall(self._NO_OP, [])]
        return env,self.get_end(env=env)

    '''
        Return agent state
    '''
    def get_state(self, env):


        state = [0, 0, 0, 0, 0,0, 0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]


        if len(self.__get_group(env, self._TERRAN_SCV)) - 12 > 0:
            state[0] = env.observation.player.idle_worker_count / (len(self.__get_group(env, self._TERRAN_SCV)) - 12)
        else:
            state[0] = 0


        
        state[1] = self.__get_percentage_workers_harvesting(env)

        percentageFood = env.observation.player.food_used / env.observation.player.food_cap
        state[2] = percentageFood

        percentageMinerals = env.observation.player.minerals / 2000
        if percentageMinerals > 1:
            state[3] = 1
        else:
            state[3] = percentageMinerals

        state[4] = self.__get_number_of_built_building(env, self._TERRAN_SUPLY_DEPOT) / self.maxHouses
        state[5] = self.__get_number_of_built_building(env, self._TERRAN_BARRACKS) / self.maxBarracks

        if self.__get_group(env, self._TERRAN_COMMANDCENTER)[0].order_progress_0 > 0:
            state[6] = 1
        else:
            state[6] = 0
        

        for i in range(7, 7 + self.__get_barracks_used(env)):
            state[i] = 1
        for i in range(15, 15 + self.__get_buildings_building(env, self._TERRAN_SUPLY_DEPOT)):
            state[i] = 1
        for i in range(23, 23 + self.__get_buildings_building(env, self._TERRAN_BARRACKS)):
            state[i] = 1
            
        return state



    '''
        Return action of environment
    '''
    def get_action(self, env, action):
        marinex, mariney = self.__get_unit_pos(env=env, view=self._PLAYER_SELF)
        func = [actions.FunctionCall(self._NO_OP, [])]
        commandCenter = self.__get_group(env, self._TERRAN_COMMANDCENTER)

        if  self.possible_actions[action] == self._IDLE_WORKER_HARVEST:
            minerals = self.__get_group(env, self._MINERAL_FIELD)
            mineral = minerals[random.randint(0, len(minerals)-1)]
            
            func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]),actions.FunctionCall(self._HARVEST_GATHER_SCREEN, [self._NOT_QUEUED, [mineral.x,mineral.y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]

        elif self.possible_actions[action] == self._IDLE_WORKER_BUILD_HOUSE:
            x = 0
            y = 0
            if (len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT)) < 11):
                x = 3.5 + (len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT)) * 6)
                y = 43.5
            else:
                x = 15.5 + ((len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT))-11) * 6)
                y = 37.5
            func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]), actions.FunctionCall(self._BUILD_SUPPLYDEPOT, [self._NOT_QUEUED, [x,y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]

            if len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT)) >= self.maxHouses:
                func = [actions.FunctionCall(self._NO_OP, [])]

        elif self.possible_actions[action] == self._IDLE_WORKER_BUILD_BARRACKS:
            func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]), actions.FunctionCall(self._BUILD_BARRACKS, [self._NOT_QUEUED, [60.5 - (len(self.__get_group(env, self._TERRAN_BARRACKS)) * 8), 5]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
            if len(self.__get_group(env, self._TERRAN_BARRACKS)) >= self.maxBarracks:
                func = [actions.FunctionCall(self._NO_OP, [])]

        elif self.possible_actions[action] == self._RANDOM_WORKER_BUILD_HOUSE:
            x = 0
            y = 0
            if (len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT)) < 11):
                x = 3.5 + (len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT)) * 6)
                y = 43.5
            else:
                x = 15.5 + ((len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT))-11) * 6)
                y = 37.5
            point = self.__get_harvesting_worker(env)
            func = [actions.FunctionCall(self._SELECT_POINT, [[0], [point[0],point[1]]]), actions.FunctionCall(self._BUILD_SUPPLYDEPOT, [self._NOT_QUEUED, [x,y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]

            if len(self.__get_group(env, self._TERRAN_SUPLY_DEPOT)) >= self.maxHouses:
                func = [actions.FunctionCall(self._NO_OP, [])]

        elif self.possible_actions[action] == self._RANDOM_WORKER_BUILD_BARRACKS:
            point = self.__get_harvesting_worker(env)
            func = [actions.FunctionCall(self._SELECT_POINT, [[0], [point[0],point[1]]]), actions.FunctionCall(self._BUILD_BARRACKS, [self._NOT_QUEUED, [60.5 - (len(self.__get_group(env, self._TERRAN_BARRACKS)) * 8), 5]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
            if len(self.__get_group(env, self._TERRAN_BARRACKS)) >= self.maxBarracks:
                func = [actions.FunctionCall(self._NO_OP, [])]

        elif self.possible_actions[action] == self._CREATE_WORKER:
            if self.__get_group(env, self._TERRAN_COMMANDCENTER)[0].order_progress_0 > 0:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]]), actions.FunctionCall(self._RALLY_WORKERS_SCREEN, [self._NOT_QUEUED, [commandCenter[0].x,commandCenter[0].y]]),actions.FunctionCall(self._TRAIN_WORKER, [self._NOT_QUEUED]),actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
        elif self.possible_actions[action] == self._CREATE_MARINES:
            barrack = self.__get_unused_barrack(env)
            if barrack[0]== -1 and barrack[1] == -1:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [barrack[0],barrack[1]]]), actions.FunctionCall(self._TRAIN_MARINE, [self._NOT_QUEUED]),actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
        elif self.possible_actions[action] == self._DO_NOTHING:
            func = [actions.FunctionCall(self._NO_OP, [])]
        
        self.action = func
    
    '''
        Return reward
    '''
    def get_reward(self, env, action):

        reward = 0
        if len(self.__get_group(env, self._TERRAN_MARINE)) > self.actualMarines:
            reward = len(self.__get_group(env, self._TERRAN_MARINE)) - self.actualMarines
            self.actualMarines = len(self.__get_group(env, self._TERRAN_MARINE))


        return reward

    '''
        Return True if we must end this episode
    '''
    def get_end(self, env):
        minerals = self.__get_group(env, self._MINERAL_FIELD)
        return not minerals

    '''
        (Private method)
        Check if current action is available. If not, use default action
    '''
    def __check_action_available(self, env, action):

        if not (action[0] in env.observation.available_actions):
            return [actions.FunctionCall(self._NO_OP, [])]
        else:
            return action

            

    '''
        (Private method)
        Return unit position
    '''
    def __get_unit_pos(self, env, view):
        ai_view = env.observation['feature_screen'][self._PLAYER_RELATIVE]
        unitys, unitxs = (ai_view == view).nonzero()
        if len(unitxs) == 0:
            unitxs = np.array([0])
        if len(unitys) == 0:
            unitys = np.array([0])
        return unitxs.mean(), unitys.mean()

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
    def __get_dist(self, env):
        marinex, mariney = self.__get_unit_pos(env, self._PLAYER_SELF)
        beaconx, beacony = self.__get_unit_pos(env, self._PLAYER_NEUTRAL)

        newDist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))
        return newDist

    '''
        (Private method)
        Return specified group
    '''
    def __get_group(self, obs, group_type):
        group = [unit for unit in obs.observation['feature_units'] 
                    if unit.unit_type == group_type]
        return group

    def __get_barracks_used(self, obs):
        barracks = self.__get_group(obs, self._TERRAN_BARRACKS)
        used = 0
        for i in range(0, len(barracks)):
            if barracks[i].order_progress_0 > 0:
                used += 1
        return used

    def __get_percentage_of_barracks_used(self, obs):
        barracks = self.__get_group(obs, self._TERRAN_BARRACKS)
        used = 0
        built = 0
        for i in range(0, len(barracks)):
            if barracks[i].order_progress_0 > 0:
                used += 1
            if barracks[i].build_progress < 100:
                built += 1
        if built == 0:
            return 0
        else:
            return used / built

    def __get_buildings_building(self, obs, buildingType):
        workers = self.__get_group(obs, self._TERRAN_SCV)
        edificios = 0
        for i in range(0, len(workers)):
            if workers[i].active == 1 and buildingType == self._TERRAN_BARRACKS and workers[i].order_id_0 == 185:
                edificios += 1
            elif workers[i].active == 1 and buildingType == self._TERRAN_SUPLY_DEPOT and workers[i].order_id_0 == 222:
                edificios += 1
        return edificios


        barracks = self.__get_group(obs, buildingType)
        built = 0
        for i in range(0, len(barracks)):
            if barracks[i].build_progress < 100:
                built += 1
        return built

    def __get_harvesting_worker(self, obs):
        workers = self.__get_group(obs, self._TERRAN_SCV)
        for i in range(0, len(workers)):
            if workers[i].active == 1 and (workers[i].order_id_0 == 359 or workers[i].order_id_0 == 362):
                return [workers[i].x, workers[i].y]
    
    def __get_percentage_workers_harvesting(self, env):
        commandCenter = self.__get_group(env, self._TERRAN_COMMANDCENTER)
        return commandCenter[0].assigned_harvesters / commandCenter[0].ideal_harvesters
        

    def __get_unused_barrack(self, obs):
        barracks = self.__get_group(obs, self._TERRAN_BARRACKS)
        for i in range(0, len(barracks)):
            if barracks[i].order_progress_0 == 0 and barracks[i].build_progress == 100:
                return [barracks[i].x, barracks[i].y]
        return [-1,-1]


    def __get_number_of_built_building(self, obs, group_type):
        buildings = self.__get_group(obs, group_type)
        built = 0
        for i in range(0, len(buildings)):
            if buildings[i].build_progress == 100:
                built += 1
        return built
