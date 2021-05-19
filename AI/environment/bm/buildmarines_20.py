import numpy as np
import math

from environment.bm.buildmarines import BuildMarines

class BuildMarines_20States(BuildMarines): 
    def __init__(self):
        super().__init__()

    '''
        Return basic information.
    '''
    def get_info(self):
        return {
            'actions' : self.possible_actions, 
             'num_states' : 20,
             'discount' : 0.99,
             'replay_mem_size' : 50_000,
             'learn_every' : 20,
             'min_replay_mem_size' : 512,
             'minibatch_size' : 256,
             'update_time' : 5,
             'max_cases' : 1024,
             'cases_to_delete' : 128,
             'hidden_nodes' : 150,
             'hidden_layer' : 2}

    '''
        Return agent state
        [IDLE WORKERS,
        FOOD,
        MINERALS,
        TRAINING WORKER,
        WORKERS HARVESTING, 
        HOUSE BUILD, 
        BARRACKS AVAILABLE ... 8, 
        HOUSE BUILDING, 
        BARRACKS BUILDING,
        HOUSE AVAILABLE,
        BARRACKS AVAILABLE,
        MARINE AVAILABLE,
        MARINE BUILDING
        ]
    '''
    def get_state(self):
        state = [0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0]

        if self.cont_steps < self.ignored_steps:
            return state

        # idle workers
        state[0] = self.obs.observation.player.idle_worker_count / 10

        # food
        remaining_food = self.obs.observation.player.food_cap - self.obs.observation.player.food_used
        state[1] = remaining_food/10

        # number of minerals harvested
        state[2] = self.obs.observation.player.minerals / 1000

        # if we are training a worker or not
        if self._get_group(group_type=self._TERRAN_COMMANDCENTER)[0].order_progress_0 > 0:
            state[3] = 1
        else:
            state[3] = 0

        # number of harvesting workers
        state[4] = self._get_workers_harvesting()/10

        # if we have a house built
        if self._get_number_of_built_building(group_type=self._TERRAN_SUPLY_DEPOT) > 0:
            state[5] = 1
        else:
            state[5] = 0

        # nodes active for each barrack used
        for i in range(6, 6 + self._get_number_of_built_building(group_type=self._TERRAN_BARRACKS) - self._get_barracks_used()):
            state[i] = 1

        # if we are building a house
        if self._get_buildings_building(group_type=self._TERRAN_SUPLY_DEPOT) > 0:
            state[14] = 1
        else:
            state[14] = 0
        
        # if we are building a barrack
        if self._get_buildings_building(group_type=self._TERRAN_BARRACKS) > 0:
            state[15] = 1
        else:
            state[15] = 0

        #if we could build a house
        if (self._get_number_of_built_building(group_type=self._TERRAN_SUPLY_DEPOT) + self._get_buildings_building(group_type=self._TERRAN_SUPLY_DEPOT)) < 16 and self.obs.observation.player.minerals >= 100:
            state[16] = 1
        else:
            state[16] = 0

        # if we could build a barrack
        if (self._get_number_of_built_building(group_type=self._TERRAN_BARRACKS) + self._get_buildings_building(group_type=self._TERRAN_BARRACKS)) < 8 and self.obs.observation.player.minerals >= 150 and self._get_number_of_built_building(group_type=self._TERRAN_SUPLY_DEPOT) > 0:
            state[17] = 1
        else:
            state[17] = 0

        # if we could create a marine
        if self._get_number_of_built_building(group_type=self._TERRAN_BARRACKS) - self._get_barracks_used() > 0 and self.obs.observation.player.minerals >= 50 and remaining_food > 0:
            state[18] = 1
        else:
            state[18] = 0         

        # if we are creating a marine
        if self._get_barracks_used() > 0:
            state[19] = 1
        else:
            state[19] = 0      
        

        return state