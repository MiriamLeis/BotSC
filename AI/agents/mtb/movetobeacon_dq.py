import numpy as np
import math

from agents.mtb.movetobeacon import MoveToBeacon

class DQMoveToBeacon(MoveToBeacon): 
    def __init__(self):
        super().__init__()

    '''
        Return basic information for Deep Q-Learning.
    '''
    def get_info(self):
        return {'actions' : self.possible_actions, 
             'num_states' : 8,
             'discount' : 0.99,
             'replay_mem_size' : 50_000,
             'learn_every' : 256,
             'min_replay_mem_size' : 256,
             'minibatch_size' : 64,
             'update_time' : 5,
             'max_cases' : 1024,
             'cases_to_delete' : 64,
             'hidden_nodes' : 25,
             'hidden_layer' : 1}

    '''
        Return agent state
    '''
    def get_state(self, env):

        marinex, mariney = super()._get_unit_pos(env=env, view=self._PLAYER_SELF)
        beaconx, beacony = super()._get_unit_pos(env=env, view=self._PLAYER_NEUTRAL)

        direction = [beaconx-marinex, beacony - mariney]
        np.linalg.norm(direction)
        
        vector_1 = [0, -1]
        angleD = super()._ang(vector_1, direction)

        if direction[0] > 0:
            angleD = 360 - angleD

        dist = super()._get_dist(env=env)
        norm = 1 - ((dist - 4) / (55 - 5))
        norm = round(norm,1)
        state = [0,0,0,0,0,0,0,0]
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

        return state