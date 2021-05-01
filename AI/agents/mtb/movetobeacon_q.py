import numpy as np
import math

from agents.mtb.movetobeacon import MoveToBeacon

class QMoveToBeacon(MoveToBeacon):
    def __init__(self):
        super().__init__()

    '''
        Return basic information for Q-Learning.
    '''
    def get_info(self):
        return{'actions' : self.possible_actions, 
             'learning_rate' : 0.2,
             'gamma' : 0.95}


    '''
        Return agent state
    '''
    def get_state(self, env):
        state = -1
        marinex, mariney = super()._get_unit_pos(env=env, view=self._PLAYER_SELF)
        beaconx, beacony = super()._get_unit_pos(env=env, view=self._PLAYER_NEUTRAL)

        direction = [beaconx - marinex, beacony - mariney]
        dist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))

        norm = 1 - ((dist - 4) / (55 - 5))
        norm = round(norm,1)

        if norm < 0.1:
            state = 0
        elif norm < 0.2:
            state = 1
        elif norm < 0.3:
            state = 2
        elif norm < 0.4:
            state = 3
        elif norm < 0.5:
            state = 4
        elif norm < 0.6:
            state = 5
        elif norm < 0.7:
            state = 6
        elif norm < 0.8:
            state = 7
        elif norm < 0.9:
            state = 8
        else:
            state = 9
        
        # angle between marine and beacon
        vector_1 = [0, -1]

        np.linalg.norm(direction)

        angleD = super()._ang(vector_1, direction)

        if direction[0] > 0:
            angleD = 360 - angleD
        if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
            state += 0
        elif angleD >= 22.5 and angleD < 67.5:
            state += 1 * 10
        elif angleD >= 67.5 and angleD < 112.5:
            state += 2 * 10
        elif angleD >= 112.5 and angleD < 157.5:
            state += 3 * 10
        elif angleD >= 157.5 and angleD < 202.5:
            state += 4 * 10
        elif angleD >= 202.5 and angleD < 247.5:
            state += 5 * 10
        elif angleD >= 247.5 and angleD < 292.5:
            state += 6 * 10
        elif angleD >= 292.5 and angleD < 337.5:
            state += 7 * 10

        return state