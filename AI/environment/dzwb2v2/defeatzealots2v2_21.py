import numpy as np
import math

from pysc2.lib import units

from environment.dzwb2v2.defeatzealots2v2 import DefeatZealots2v2

class DefeatZealots2v2_21States(DefeatZealots2v2): 
    def __init__(self):
        super().__init__()

    '''
        Return basic information for Deep Q-Learning.
    '''
    def get_info(self):
        if self.current_agent == 1:
            return {'actions' : self.possible_actions, 
                'num_states' : 21,
                'discount' : 0.99,
                'replay_mem_size' : 50_000,
                'learn_every' : 100,
                'min_replay_mem_size' : 500,
                'minibatch_size' : 128,
                'update_time' : 150,
                'max_cases' : 1_500,
                'cases_to_delete' : 100,
                'hidden_nodes' : 100,
                'hidden_layer' : 2}

        else:
            return {'actions' : self.possible_actions, 
                'num_states' : 21,
                'discount' : 0.99,
                'replay_mem_size' : 50_000,
                'learn_every' : 100,
                'min_replay_mem_size' : 500,
                'minibatch_size' : 128,
                'update_time' : 150,
                'max_cases' : 1_500,
                'cases_to_delete' : 100,
                'hidden_nodes' : 100,
                'hidden_layer' : 2}

    '''
        Return agent state
        state:
        [UP, UP LEFT, LEFT, DOWN LEFT, DOWN, DOWN RIGHT, RIGHT, UP RIGHT, ------> enemy position
        UP WALL, LEFT WALL, DOWN WALL, RIGHT WALL,
        COOLDOWN,
        UP, UP LEFT, LEFT, DOWN LEFT, DOWN, DOWN RIGHT, RIGHT, UP RIGHT] ------> ally position
    '''
    def get_state(self):
        # prepare state
        state = [10,10,10,10,10,10,10,10, 0,0,0,0, 0, 0,0,0,0,0,0,0,0]

        if self.current_agent == 1:
            agent_x, agent_y = super()._get_meangroup_position(group_type=self.UNIT_AGENT_1)
        else:
            agent_x, agent_y = super()._get_meangroup_position(group_type=self.UNIT_AGENT_2)

        zealots = super()._get_group(group_type=self.UNIT_ENEMY)

        for unit in zealots:
            # get direction
            direction = [unit.x - agent_x, unit.y - agent_y]
            np.linalg.norm(direction)

            vector_1 = [0, -1]
            angleD = super()._ang(vector_1, direction)

            if direction[0] > 0:
                angleD = 360 - angleD

            # check dist
            dist = super()._get_dist([agent_x, agent_y], [unit.x, unit.y])

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
        if (agent_y - self._MOVE_VAL) < 3.5:
            state[8] = 1
        if (agent_x - self._MOVE_VAL) < 3.5:
            state[9] = 1
        if (agent_y + self._MOVE_VAL) > 44.5:
            state[10] = 1
        if (agent_x + self._MOVE_VAL) > 60.5:
            state[11] = 1

        # check shoot cooldown
        if self.current_agent == 1:
            self.current_can_shoot_1 = super()._can_shoot(group_type=self.UNIT_AGENT_1)
            if self.current_can_shoot_1:
                state[12] = 1

        else:
            self.current_can_shoot_2 = super()._can_shoot(group_type=self.UNIT_AGENT_2)
            if self.current_can_shoot_2:
                state[12] = 1

        #check allies
        if self.current_agent == 1:
            ally_type = self.UNIT_AGENT_2
        else:
            ally_type = self.UNIT_AGENT_1

        allies = super()._get_group(group_type=ally_type)
        if not allies: 
            return state

        ally = allies[0]
            
            # get direction
        direction = [ally.x - agent_x, ally.y - agent_y]
        np.linalg.norm(direction)

        vector_1 = [0, -1]
        angleD = super()._ang(vector_1, direction)

        if direction[0] > 0:
            angleD = 360 - angleD

            # check dist
        dist = super()._get_dist([agent_x, agent_y], [ally.x, ally.y])

        norm = 1 - ((dist - 4) / (55 - 5))
        norm = round(norm, 1)

            # check angle
        if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
            if state[13] > norm:
                state[13] = norm
        elif angleD >= 22.5 and angleD < 67.5:
            if state[14] > norm:
                state[14] = norm
        elif angleD >= 67.5 and angleD < 112.5:
            if state[15] > norm:
                state[15] = norm
        elif angleD >= 112.5 and angleD < 157.5:
            if state[16] > norm:
                state[16] = norm
        elif angleD >= 157.5 and angleD < 202.5:
            if state[17] > norm:
                state[17] = norm
        elif angleD >= 202.5 and angleD < 247.5:
            if state[18] > norm:
                state[18] = norm
        elif angleD >= 247.5 and angleD < 292.5:
            if state[19] > norm:
                state[19] = norm
        elif angleD >= 292.5 and angleD < 337.5:
            if state[20] > norm:
                state[20] = norm

        return state