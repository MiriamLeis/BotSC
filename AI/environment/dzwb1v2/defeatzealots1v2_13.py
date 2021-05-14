import numpy as np
import math

from environment.dzwb.defeatzealots import DefeatZealots

class DefeatZealots1v2_13States(DefeatZealots): 
    def __init__(self):
        super().__init__(args=['DefeatZealotswithBlink_2enemies'])
        
    '''
        Return map information.
    '''
    def get_args(self):
        super().get_args()

        return ['DefeatZealotswithBlink_2enemies']

    '''
        Return basic information for Deep Q-Learning.
    '''
    def get_info(self):
        return {'actions' : self.possible_actions, 
             'num_states' : 13,
             'discount' : 0.99,
             'replay_mem_size' : 50_000,
             'learn_every' : 80,
             'min_replay_mem_size' : 150,
             'minibatch_size' : 64,
             'update_time' : 50,
             'max_cases' : 1_000,
             'cases_to_delete' : 100,
             'hidden_nodes' : 100,
             'hidden_layer' : 2}

    '''
        Return agent state
        state:
        [UP, UP LEFT, LEFT, DOWN LEFT, DOWN, DOWN RIGHT, RIGHT, UP RIGHT, ------> enemy position
        UP WALL, LEFT WALL, DOWN WALL, RIGHT WALL,
        COOLDOWN]
    '''
    def get_state(self):
        # prepare state
        state = [10,10,10,10,10,10,10,10, 0, 0,0,0,0]

        stalkerx, stalkery = super()._get_meangroup_position(group_type=self.UNIT_ALLY)
        zealots = super()._get_group(group_type=self.UNIT_ENEMY)

        for unit in zealots:
            # get direction
            direction = [unit.x - stalkerx, unit.y - stalkery]
            np.linalg.norm(direction)

            vector_1 = [0, -1]
            angleD = super()._ang(vector_1, direction)

            if direction[0] > 0:
                angleD = 360 - angleD

            # check dist
            dist = super()._get_dist([stalkerx, stalkery], [unit.x, unit.y])

            norm = 1 - ((dist - 4) / (55 - 5))
            norm = round(norm, 1)
            # check angle
            if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
                if state[0] > norm:
                    state[0] = norm
            elif angleD >= 22.5 and angleD < 67.5:
                if state[1] > norm:
                    state[1] = norm
            elif angleD >= 67.5 and angleD < 112.5:
                if state[2] > norm:
                    state[2] = norm
            elif angleD >= 112.5 and angleD < 157.5:
                if state[3] > norm:
                    state[3] = norm
            elif angleD >= 157.5 and angleD < 202.5:
                if state[4] > norm:
                    state[4] = norm
            elif angleD >= 202.5 and angleD < 247.5:
                if state[5] > norm:
                    state[5] = norm
            elif angleD >= 247.5 and angleD < 292.5:
                if state[6] > norm:
                    state[6] = norm
            elif angleD >= 292.5 and angleD < 337.5:
                if state[7] > norm:
                    state[7] = norm

        for i in range(8):
            if state[i] == 10:
                state[i] = 0

        # check limits
        if (stalkery - self._MOVE_VAL) < 3.5:
            state[8] = 1
        if (stalkerx - self._MOVE_VAL) < 3.5:
            state[9] = 1
        if (stalkery + self._MOVE_VAL) > 44.5:
            state[10] = 1
        if (stalkerx + self._MOVE_VAL) > 60.5:
            state[11] = 1

        # check cooldown
        self.current_can_shoot = False
        if super()._can_shoot(group_type=self.UNIT_ALLY):
            state[12] = 1
            self.current_can_shoot = True

        return state