import numpy as np
import math
import random 
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

from environment.starcraft_env import StarcraftEnv # environment

class BuildMarines(StarcraftEnv): 
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
        super().__init__(args=['BuildMarines'])

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

        self.marines_max = 159

    '''
        Prepare basic parameters.
    '''
    def prepare(self):
        self.houses_build = 0
        self.maxHouses = 20

        self.barracks_build = 0
        self.maxBarracks = 8

        self.actualMarines = 0
        self.could_create_marine = False

        self.ignored_steps = 25
        self.cont_steps = 0

        self.action = actions.FunctionCall(self._NO_OP, [])

    '''
        Update basic values and train
    '''
    def update(self, deltaTime):
        self.cont_steps += 1

    '''
        Do step of the environment
    '''
    def step(self):
        for func in self.action:
            finalAct = self.__check_action_available(action=func)
            obs = self.env.step(actions=[finalAct])
            self.obs = obs[0]
        self.action = [actions.FunctionCall(self._NO_OP, [])]

    '''
        Return action of environment
    '''
    def get_action(self, action):
        func = [actions.FunctionCall(self._NO_OP, [])]

        if self.cont_steps < self.ignored_steps: 
            self.action = func
            return

        commandCenter = self._get_group(group_type=self._TERRAN_COMMANDCENTER)

        # harvest with idle worker
        if  self.possible_actions[action] == self._IDLE_WORKER_HARVEST:
            if self._get_workers_harvesting() >= 16:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                minerals = self._get_group(group_type=self._MINERAL_FIELD)
                mineral = minerals[random.randint(0, len(minerals)-1)]

                func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]),actions.FunctionCall(self._HARVEST_GATHER_SCREEN, [self._NOT_QUEUED, [mineral.x,mineral.y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]

        # build house with idle worker
        elif self.possible_actions[action] == self._IDLE_WORKER_BUILD_HOUSE:
            self.houses_build, func = self.__build_with_idle_worker(
                                                                minerals_limit=100,
                                                                max_number=self.maxHouses, 
                                                                current_built=self.houses_build, 
                                                                building_positions=self.houses_positions, 
                                                                build_action_type=self._BUILD_SUPPLYDEPOT, 
                                                                build_type=self._TERRAN_SUPLY_DEPOT, 
                                                                other_build_type=None)

        # build barrack with idle worker
        elif self.possible_actions[action] == self._IDLE_WORKER_BUILD_BARRACKS:
            self.barracks_build, func = self.__build_with_idle_worker( 
                                                                minerals_limit=150,
                                                                max_number=self.maxBarracks, 
                                                                current_built=self.barracks_build, 
                                                                building_positions=self.barracks_positions, 
                                                                build_action_type=self._BUILD_BARRACKS, 
                                                                build_type=self._TERRAN_BARRACKS, 
                                                                other_build_type=self._TERRAN_SUPLY_DEPOT)

        # build house with worker
        elif self.possible_actions[action] == self._RANDOM_WORKER_BUILD_HOUSE:
            self.houses_build, func = self.__build_with_random_worker( 
                                                                minerals_limit=100,
                                                                max_number=self.maxHouses, 
                                                                current_built=self.houses_build, 
                                                                building_positions=self.houses_positions, 
                                                                build_action_type=self._BUILD_SUPPLYDEPOT, 
                                                                build_type=self._TERRAN_SUPLY_DEPOT, 
                                                                other_build_type=None)

        # build barrack with worker
        elif self.possible_actions[action] == self._RANDOM_WORKER_BUILD_BARRACKS:
            self.barracks_build, func = self.__build_with_random_worker(
                                                                minerals_limit=150,
                                                                max_number=self.maxBarracks, 
                                                                current_built=self.barracks_build, 
                                                                building_positions=self.barracks_positions, 
                                                                build_action_type=self._BUILD_BARRACKS, 
                                                                build_type=self._TERRAN_BARRACKS, 
                                                                other_build_type=self._TERRAN_SUPLY_DEPOT)

        # create worker
        elif self.possible_actions[action] == self._CREATE_WORKER:
            if self._get_group(group_type=self._TERRAN_COMMANDCENTER)[0].order_progress_0 > 0:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]]), actions.FunctionCall(self._RALLY_WORKERS_SCREEN, [self._NOT_QUEUED, [commandCenter[0].x,commandCenter[0].y]]),actions.FunctionCall(self._TRAIN_WORKER, [self._NOT_QUEUED]),actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
        
        # create marine
        elif self.possible_actions[action] == self._CREATE_MARINES:
            barrack = self._get_unused_barrack()
            if barrack == None:
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
    def get_reward(self, action):
        reward = 0

        if len(self._get_group(group_type=self._TERRAN_MARINE)) > self.actualMarines:
            reward = len(self._get_group(group_type=self._TERRAN_MARINE)) - self.actualMarines
            self.actualMarines = len(self._get_group(group_type=self._TERRAN_MARINE))
        
        if (self.possible_actions[action] == self._CREATE_MARINES) and not self.could_create_marine:
            reward -= 1
            
        return reward

    '''
        Return True if we must end this episode
    '''
    def get_end(self):
        minerals = self._get_group(group_type=self._MINERAL_FIELD)
        ideal = self._get_ideal_harvesting()
        return not minerals or (ideal != 16)

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
        Return number of barracks used
    '''    
    def _get_barracks_used(self):
        barracks = self._get_group(group_type=self._TERRAN_BARRACKS)
        used = 0
        for i in range(0, len(barracks)):
            if barracks[i].order_progress_0 > 0:
                used += 1
        return used
        
    '''
        (Protected method)
        Return percentage of barracks used
    '''    
    def _get_percentage_of_barracks_used(self):
        barracks = self._get_group(group_type=self._TERRAN_BARRACKS)
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
        
    '''
        (Protected method)
        Return position of unused barrack.
        In case of not found unused barrack, return None
    '''
    def _get_unused_barrack(self):
        barracks = self._get_group(group_type=self._TERRAN_BARRACKS)
        for i in range(0, len(barracks)):
            if barracks[i].order_progress_0 == 0 and barracks[i].build_progress == 100:
                return [barracks[i].x, barracks[i].y]
        return None
        
    '''
        (Protected method)
        Return number of specify building buildings
    '''    
    def _get_buildings_building(self, group_type):
        workers = self._get_group(group_type=self._TERRAN_SCV)
        edificios = 0
        for i in range(0, len(workers)):
            if workers[i].active == 1 and group_type == self._TERRAN_BARRACKS and workers[i].order_id_0 == 185:
                edificios += 1
            elif workers[i].active == 1 and group_type == self._TERRAN_SUPLY_DEPOT and workers[i].order_id_0 == 222:
                edificios += 1
        return edificios


    '''
        (Protected method)
        Return number of specify built building
    '''
    def _get_number_of_built_building(self, group_type):
        buildings = self._get_group(group_type=group_type)
        built = 0
        for i in range(0, len(buildings)):
            if buildings[i].build_progress == 100:
                built += 1
        return built
        
    '''
        (Protected method)
        Return a harvesting worker.
        In case of not exist harvesting worker, return None
    '''    
    def _get_harvesting_worker(self):
        workers = self._get_group(group_type=self._TERRAN_SCV)
        for i in range(0, len(workers)):
            if workers[i].active == 1 and (workers[i].order_id_0 == 359 or workers[i].order_id_0 == 362):
                return [workers[i].x, workers[i].y]
        return None
        
    '''
        (Protected method)
        Return current number of harvesting workers
    '''    
    def _get_workers_harvesting(self):
        commandCenter = self._get_group(group_type=self._TERRAN_COMMANDCENTER)
        return commandCenter[0].assigned_harvesters

    '''
        (Protected method)
        Return idel number of harvesting workers
    '''    
    def _get_ideal_harvesting(self):
        commandCenter = self._get_group(group_type=self._TERRAN_COMMANDCENTER)
        return commandCenter[0].ideal_harvesters

    '''
        (Private method)
        Check if current action is available. If not, use default action
    '''
    def __check_action_available(self, action):
        if not (action[0] in self.obs.observation.available_actions):
            return [actions.FunctionCall(self._NO_OP, [])]
        else:
            return action

    '''
        (Private method)
        Build specify building with idle worker if it is possible. 
        Return current number of that kind of building and action function
    '''
    def __build_with_idle_worker(self, minerals_limit, max_number, current_built, building_positions, build_action_type, build_type, other_build_type = None):
        commandCenter = self._get_group(group_type=self._TERRAN_COMMANDCENTER)

        buildings_number = len(self._get_group(group_type=build_type)) + self._get_buildings_building(group_type=build_type)
        if current_built >= max_number and buildings_number < max_number:
            lost_buildings = self._get_group(group_type=build_type)
            positions = building_positions.copy()
            for building in lost_buildings:
                for pos in building_positions:
                    x = building.x
                    y = building.y
                    if (building.x >= pos['x'] - 1.5 and building.x <= pos['x'] + 1.5) and (building.y >= pos['y'] - 1.5 and building.y <= pos['y'] + 1.5):
                        positions.remove(pos)
                        break

            func = [actions.FunctionCall(self._SELECT_IDLE_WORKER, [[0]]), actions.FunctionCall(build_action_type, [self._NOT_QUEUED, [positions[0]['x'], positions[0]['y']]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]

        elif current_built >= max_number or self.obs.observation.player.idle_worker_count <= 0 or self.obs.observation.player.minerals < minerals_limit or (other_build_type != None and self._get_number_of_built_building(group_type=other_build_type) <= 0):
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
    def __build_with_random_worker(self, minerals_limit, max_number, current_built, building_positions, build_action_type, build_type, other_build_type = None):
        commandCenter = self._get_group(group_type=self._TERRAN_COMMANDCENTER)

        buildings_number = len(self._get_group(group_type=build_type)) + self._get_buildings_building(group_type=build_type)
        if current_built >= max_number and buildings_number < max_number:
            lost_building = self._get_group(group_type=build_type)
            positions = building_positions.copy()
            for building in lost_building:
                for pos in building_positions:
                    x = building.x
                    y = building.y
                    if (building.x >= pos['x'] - 1.5 and building.x <= pos['x'] + 1.5) and (building.y >= pos['y'] - 1.5 and building.y <= pos['y'] + 1.5):
                        positions.remove(pos)
                        break

            point = self._get_harvesting_worker()
            if point == None:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [point[0],point[1]]]), actions.FunctionCall(build_action_type, [self._NOT_QUEUED, [positions[0]['x'], positions[0]['y']]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]          

        elif current_built >= max_number or self.obs.observation.player.minerals < minerals_limit or (other_build_type != None and self._get_number_of_built_building(group_type=other_build_type) <= 0):
            func = [actions.FunctionCall(self._NO_OP, [])]

        else:
            x = building_positions[current_built]["x"]
            y = building_positions[current_built]["y"]

            point = self._get_harvesting_worker()
            if point == None:
                func = [actions.FunctionCall(self._NO_OP, [])]
            else:
                func = [actions.FunctionCall(self._SELECT_POINT, [[0], [point[0],point[1]]]), actions.FunctionCall(build_action_type, [self._NOT_QUEUED, [x, y]]), actions.FunctionCall(self._SELECT_POINT, [[0], [commandCenter[0].x,commandCenter[0].y]])]
                current_built += 1

        return current_built, func
