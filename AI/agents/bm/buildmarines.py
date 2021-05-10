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

        self.houses_positions =[
                {'x': 3.5, 'y': 43.5},
                {'x': 3.5 + 6, 'y': 43.5},
                {'x': 3.5 + 12, 'y': 43.5},
                {'x': 3.5 + 18, 'y': 43.5},
                {'x': 3.5 + 24, 'y': 43.5},
                {'x': 3.5 + 30, 'y': 43.5},
                {'x': 3.5 + 36, 'y': 43.5},
                {'x': 3.5 + 42, 'y': 43.5},
                {'x': 3.5 + 48, 'y': 43.5},
                {'x': 3.5 + 54, 'y': 43.5},
                {'x': 3.5 + 60, 'y': 43.5},

                {'x': 15.5, 'y': 37.5},
                {'x': 15.5 + 6, 'y': 37.5},
                {'x': 15.5 + 12, 'y': 37.5},
                {'x': 15.5 + 18, 'y': 37.5},
                {'x': 15.5 + 24, 'y': 37.5},
                {'x': 15.5 + 30, 'y': 37.5},
                {'x': 15.5 + 36, 'y': 37.5},
                {'x': 15.5 + 42, 'y': 37.5},
                {'x': 15.5 + 48, 'y': 37.5},
        ]

        self.barracks_positions =[
                {'x': 60.5, 'y': 5},
                {'x': 60.5 - 8, 'y': 5},
                {'x': 60.5 - 16, 'y': 5},
                {'x': 60.5 - 24, 'y': 5},
                {'x': 60.5 - 32, 'y': 5},
                {'x': 60.5 - 40, 'y': 5},
                {'x': 60.5 - 48, 'y': 5},
                {'x': 60.5 - 56, 'y': 5}
        ]

    def get_args(self):
        super().get_args()

        return ['BuildMarines']
    
    '''
        Return basic information.
    '''
    def get_info(self):
        return {
            'actions' : self.possible_actions, 
             'num_states' : 68,
             'discount' : 0.99,
             'replay_mem_size' : 50_000,
             'learn_every' : 20,
             'min_replay_mem_size' : 512,
             'minibatch_size' : 256,
             'update_time' : 5,
             'max_cases' : 1024,
             'cases_to_delete' : 128,
             'hidden_nodes' : 100,
             'hidden_layer' : 2}

    '''
        Prepare basic parameters.
    '''
    def prepare(self, env):
        super().prepare(env=env)

        self.houses_build = 0
        self.maxHouses = 20

        self.barracks_build = 0
        self.maxBarracks = 8

        self.actualMarines = 0

        self.ignored_steps = 25
        self.cont_steps = 0

        self.action = actions.FunctionCall(self._NO_OP, [])

        return 0

    '''
        Update basic values and train
    '''
    def update(self, env, deltaTime):
        super().update(env=env, deltaTime=deltaTime)
        self.cont_steps += 1

    '''
        Do step of the environment
    '''
    def step(self, env, environment):
        for func in self.action:
            finalAct = self.__check_action_available(env=env, action=func)
            env = environment.step(actions=[finalAct])
        self.action = [actions.FunctionCall(self._NO_OP, [])]
        return env

    '''
        Return agent state
        [PERCENTAGE IDLE WORKERS,
        PERCENTAGE FOOD,
        PERCENTAGE MINERALS,
        TRAINING WORKER,
        WORKER 1 HARVESTING, ..., WORKER 16 HARVESTING, 
        HOUSE 1 BUILD, ..., HOUSE 16 BUILD, 
        BARRACK 1 BUILD, ..., BARRACK 8 BUILD, 
        BARRACK 1 USED, ... , BARRACK 8 USED, 
        HOUSE 1 BUILDING, ... , HOUSE 8 BUILDING, 
        BARRACKS 1 BUILDING, ... , BARRACKS 8 BUILDING, ]
    '''
    def get_state(self, env):
        state = [0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]

        if self.cont_steps < self.ignored_steps: return state

        if len(self.__get_group(env, self._TERRAN_SCV)) - 12 > 0:
            state[0] = round(env.observation.player.idle_worker_count / (len(self.__get_group(env, self._TERRAN_SCV)) - 12), 2)
        else:
            state[0] = 0

        percentageFood = env.observation.player.food_used / env.observation.player.food_cap
        state[1] = round(percentageFood, 2)

        percentageMinerals = env.observation.player.minerals / 2000
        if percentageMinerals > 1:
            state[2] = 1
        else:
            state[2] = round(percentageMinerals, 2)

        if self.__get_group(env, self._TERRAN_COMMANDCENTER)[0].order_progress_0 > 0:
            state[3] = 1
        else:
            state[3] = 0

        
        for i in range(4, 4 + self.__get_workers_harvesting(env)):
            state[i] = 1

        for i in range(20, 20 + self.__get_number_of_built_building(env, self._TERRAN_SUPLY_DEPOT)):
            state[i] = 1
        
        for i in range(36, 36 + self.__get_number_of_built_building(env, self._TERRAN_BARRACKS)):
            state[i] = 1

        for i in range(44, 44 + self.__get_barracks_used(env)):
            state[i] = 1

        for i in range(52, 52 + self.__get_buildings_building(env, self._TERRAN_SUPLY_DEPOT)):
            state[i] = 1

        for i in range(60, 60 + self.__get_buildings_building(env, self._TERRAN_BARRACKS)):
            state[i] = 1
            
        return state



    '''
        Return action of environment
    '''
    def get_action(self, env, action):
        func = [actions.FunctionCall(self._NO_OP, [])]

        if self.cont_steps < self.ignored_steps: 
            self.action = func
            return

        commandCenter = self.__get_group(env, self._TERRAN_COMMANDCENTER)

        # harvest with idle worker
        if  self.possible_actions[action] == self._IDLE_WORKER_HARVEST:
            if self.__get_workers_harvesting(env) >= 16:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                minerals = self.__get_group(env, self._MINERAL_FIELD)
                mineral = minerals[random.randint(0, len(minerals)-1)]

                func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]),actions.FunctionCall(self._HARVEST_GATHER_SCREEN, [self._NOT_QUEUED, [mineral.x,mineral.y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]

        # build house with idle worker
        elif self.possible_actions[action] == self._IDLE_WORKER_BUILD_HOUSE:
            self.houses_build, func = self.__build_with_idle_worker(env=env, 
                                                                minerals_limit=100,
                                                                max_number=self.maxHouses, 
                                                                current_built=self.houses_build, 
                                                                building_positions=self.houses_positions, 
                                                                build_action_type=self._BUILD_SUPPLYDEPOT, 
                                                                build_type=self._TERRAN_SUPLY_DEPOT, 
                                                                other_build_type=None)

        # build barrack with idle worker
        elif self.possible_actions[action] == self._IDLE_WORKER_BUILD_BARRACKS:
            self.barracks_build, func = self.__build_with_idle_worker(env=env, 
                                                                minerals_limit=150,
                                                                max_number=self.maxBarracks, 
                                                                current_built=self.barracks_build, 
                                                                building_positions=self.barracks_positions, 
                                                                build_action_type=self._BUILD_BARRACKS, 
                                                                build_type=self._TERRAN_BARRACKS, 
                                                                other_build_type=self._TERRAN_SUPLY_DEPOT)

        # build house with worker
        elif self.possible_actions[action] == self._RANDOM_WORKER_BUILD_HOUSE:
            self.houses_build, func = self.__build_with_random_worker(env=env, 
                                                                minerals_limit=100,
                                                                max_number=self.maxHouses, 
                                                                current_built=self.houses_build, 
                                                                building_positions=self.houses_positions, 
                                                                build_action_type=self._BUILD_SUPPLYDEPOT, 
                                                                build_type=self._TERRAN_SUPLY_DEPOT, 
                                                                other_build_type=None)

        # build barrack with worker
        elif self.possible_actions[action] == self._RANDOM_WORKER_BUILD_BARRACKS:
            self.barracks_build, func = self.__build_with_random_worker(env=env, 
                                                                minerals_limit=150,
                                                                max_number=self.maxBarracks, 
                                                                current_built=self.barracks_build, 
                                                                building_positions=self.barracks_positions, 
                                                                build_action_type=self._BUILD_BARRACKS, 
                                                                build_type=self._TERRAN_BARRACKS, 
                                                                other_build_type=self._TERRAN_SUPLY_DEPOT)

        # create worker
        elif self.possible_actions[action] == self._CREATE_WORKER:
            if self.__get_group(env, self._TERRAN_COMMANDCENTER)[0].order_progress_0 > 0:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]]), actions.FunctionCall(self._RALLY_WORKERS_SCREEN, [self._NOT_QUEUED, [commandCenter[0].x,commandCenter[0].y]]),actions.FunctionCall(self._TRAIN_WORKER, [self._NOT_QUEUED]),actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
        
        # create marine
        elif self.possible_actions[action] == self._CREATE_MARINES:
            barrack = self.__get_unused_barrack(env)
            if barrack[0]== -1 and barrack[1] == -1:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [barrack[0],barrack[1]]]), actions.FunctionCall(self._TRAIN_MARINE, [self._NOT_QUEUED]),actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
        
        # do nothing
        elif self.possible_actions[action] == self._DO_NOTHING:
            func = [actions.FunctionCall(self._NO_OP, [])]
        
        self.action = func
    
    '''
        Return reward
    '''
    def get_reward(self, env, action):
        reward = 0
        if len(self.__get_group(env, self._TERRAN_MARINE)) > self.actualMarines:
            reward = self.actualMarines
            #reward = len(self.__get_group(env, self._TERRAN_MARINE)) - self.actualMarines
            self.actualMarines = len(self.__get_group(env, self._TERRAN_MARINE))


        return reward

    '''
        Return True if we must end this episode
    '''
    def get_end(self, env):
        minerals = self.__get_group(env, self._MINERAL_FIELD)
        ideal = self.__get_ideal_harvesting(env)
        return not minerals or (ideal != 16)

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
        Build specify building with idle worker if it is possible. 
        Return current number of that kind of building and action function
    '''
    def __build_with_idle_worker(self, env, minerals_limit, max_number, current_built, building_positions, build_action_type, build_type, other_build_type = None):
        commandCenter = self.__get_group(env, self._TERRAN_COMMANDCENTER)

        buildings_number = len(self.__get_group(env, build_type)) + self.__get_buildings_building(env, build_type)
        if current_built >= max_number and buildings_number < max_number:
            lost_buildings = self.__get_group(env, build_type)
            positions = building_positions.copy()
            for building in lost_buildings:
                for pos in building_positions:
                    x = building.x
                    y = building.y
                    if (building.x >= pos['x'] - 1.5 and building.x <= pos['x'] + 1.5) and (building.y >= pos['y'] - 1.5 and building.y <= pos['y'] + 1.5):
                        positions.remove(pos)
                        break

            func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]), actions.FunctionCall(build_action_type, [self._NOT_QUEUED, [positions[0]['x'], positions[0]['y']]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]

        elif current_built >= max_number or env.observation.player.idle_worker_count <= 0 or env.observation.player.minerals < minerals_limit or (other_build_type != None and self.__get_number_of_built_building(env, other_build_type) <= 0):
            func = [actions.FunctionCall(self._NO_OP, [])]

        else:
            x = building_positions[current_built]["x"]
            y = building_positions[current_built]["y"]

            func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]), actions.FunctionCall(build_action_type, [self._NOT_QUEUED, [x, y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
            current_built += 1
        
        return current_built, func

    '''
        (Private method)
        Build specify building with random worker if it is possible. 
        Return current number of that kind of building and action function
    '''
    def __build_with_random_worker(self, env, minerals_limit, max_number, current_built, building_positions, build_action_type, build_type, other_build_type = None):
        commandCenter = self.__get_group(env, self._TERRAN_COMMANDCENTER)

        buildings_number = len(self.__get_group(env, build_type)) + self.__get_buildings_building(env, build_type)
        if current_built >= max_number and buildings_number < max_number:
            lost_building = self.__get_group(env, build_type)
            positions = building_positions.copy()
            for building in lost_building:
                for pos in building_positions:
                    x = building.x
                    y = building.y
                    if (building.x >= pos['x'] - 1.5 and building.x <= pos['x'] + 1.5) and (building.y >= pos['y'] - 1.5 and building.y <= pos['y'] + 1.5):
                        positions.remove(pos)
                        break

            point = self.__get_harvesting_worker(env)
            if point == None:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [point[0],point[1]]]), actions.FunctionCall(build_action_type, [self._NOT_QUEUED, [positions[0]['x'], positions[0]['y']]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]          

        elif current_built >= max_number or env.observation.player.minerals < minerals_limit or (other_build_type != None and self.__get_number_of_built_building(env, other_build_type) <= 0):
            func = [actions.FunctionCall(self._NO_OP, [])]

        else:
            x = building_positions[current_built]["x"]
            y = building_positions[current_built]["y"]

            point = self.__get_harvesting_worker(env)
            if point == None:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [point[0],point[1]]]), actions.FunctionCall(build_action_type, [self._NOT_QUEUED, [x, y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
                current_built += 1

        return current_built, func

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
    
    def __get_workers_harvesting(self, env):
        commandCenter = self.__get_group(env, self._TERRAN_COMMANDCENTER)
        return commandCenter[0].assigned_harvesters
    
    def __get_ideal_harvesting(self, env):
        commandCenter = self.__get_group(env, self._TERRAN_COMMANDCENTER)
        return commandCenter[0].ideal_harvesters
        

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
