import math

import numpy as np
import pandas as pd

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents import base_agent

# constantes para las caracteristicas
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1  # jugador
_PLAYER_NEUTRAL = 3  # beacon

# constantes para las acciones
_QUEUED = [0]
_NOT_QUEUED = [0]

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id

ACTION_MOVE_NORTH = 0
ACTION_MOVE_EAST = 1
ACTION_MOVE_SOUTH = 2
ACTION_MOVE_WEST = 3

our_actions = [
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_EAST,
    ACTION_MOVE_WEST
]

# constantes para las recompensas
APPROACHING_REWARD = 0.5
LEAVING_REWARD = -0.5

# constantes del juego
MOVE_VALUE = 2.0
WORLD_WIDTH = 64
WORLD_HEIGHT = 64


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.8):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = np.zeros((0, len(our_actions)))
        self.states_list = set()
        self.load_qtable('agent3_qtable.npy')
        self.load_states('agent3_states.npy')
    def __del__(self):
        self.save_qtable('agent3_qtable.npy')
        self.save_states('agent3_states.npy')

    def choose_action(self, state):
        self.check_state_exist(state)

        if np.random.uniform() <= self.epsilon:
            idx = list(self.states_list).index(state)
            q_values = self.q_table[idx]
            action = int(np.argmax(q_values))

        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        self.check_state_exist(state)

        state_idx = list(self.states_list).index(state)
        next_state_idx = list(self.states_list).index(state_)

        q_predict = self.q_table[state_idx, action]

        # almacena la recompensa por:
        # la accion que ha realizado + lo bueno que es el estado al que te ha llevado
        q_target = reward + self.gamma * self.q_table[next_state_idx].max()

        # actualizar la recompensa de ese estado con esa accion dependiendo de lo que hubiese antes
        #self.q_table[state_idx, action] = (1 - self.lr) * q_predict + self.lr * (q_target)
        self.q_table[state_idx, action] += self.lr * (q_target - q_predict)
        

    def check_state_exist(self, state):
        if state not in self.states_list:
            self.q_table = np.vstack([self.q_table, np.zeros((1, len(our_actions)))])
            self.states_list.add(state)

    def save_qtable(self, filepath):
        np.save(filepath, self.q_table)

    def load_qtable(self, filepath):
        try:
            self.q_table = np.load(filepath)
        except:
            pass

    def save_states(self, filepath):
        temp = np.array(list(self.states_list))
        np.save(filepath, temp)

    def load_states(self, filepath):
        try:
            temp = self.load_states(load_st)
            self.states_list = set([tuple(temp[i]) for i in range(len(temp))])
        except:
            pass

def get_player_location(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    return (ai_view == _PLAYER_SELF).nonzero()


def get_beacon_location(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    return (ai_view == _PLAYER_NEUTRAL).nonzero()



class MoveToBeaconAgent(base_agent.BaseAgent):
    def __init__(self):
        super(MoveToBeaconAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(our_actions))))

        self.previous_action = None
        self.previous_state = None


    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)

        # si podemos mover nuestro ejercito (si podemos mover algo)
        if _MOVE_SCREEN in obs.observation['available_actions']:
            # conseguimos las posiciones que nos interesan
            player_xs, player_ys = get_player_location(obs)
            beacon_xs, beacon_ys = get_beacon_location(obs)

            if len(player_xs) == 0:
                player_xs = np.array([0])
            if len(player_ys) == 0:
                player_ys = np.array([0])
            player_position = [player_xs.mean(), player_ys.mean()]
            beacon_position = [beacon_xs.mean(), beacon_ys.mean()]
            # creamos el estado actual
            current_state = [
                player_position,
                beacon_position
            ]
            reward = 0
            # dar recompensas a la accion anterior
            if self.previous_action is not None:


                prev_player_pos = self.previous_state[0]
                prev_beacon_pos = self.previous_state[1]


                previous_dist = math.sqrt(pow(prev_player_pos[0] - prev_beacon_pos[0], 2) +
                                          pow(prev_player_pos[1] - prev_beacon_pos[1], 2))
                actual_dist = math.sqrt(pow(player_position[0] - beacon_position[0], 2) +
                                        pow(player_position[1] - beacon_position[1], 2))

                if previous_dist > actual_dist:
                    reward += APPROACHING_REWARD
                else:
                    reward += LEAVING_REWARD

                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

            # actualizar estado y accion
            action = self.qlearn.choose_action(str(current_state))
            actual_action = our_actions[action]

            self.previous_state = current_state
            self.previous_action = action
            # hacer la accion
            if actual_action == ACTION_MOVE_EAST:
                target = [player_position[1] + MOVE_VALUE, player_position[0]]

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            elif actual_action == ACTION_MOVE_WEST:
                target = [player_position[1] - MOVE_VALUE, player_position[0]]

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            elif actual_action == ACTION_MOVE_SOUTH:
                target = [player_position[1], player_position[0] + MOVE_VALUE]

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            elif actual_action == ACTION_MOVE_NORTH:
                target = [player_position[1], player_position[0] - MOVE_VALUE]

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            return actions.FunctionCall(_NO_OP, [])

        # si no podemos movernos es porque no tenemos nada seleccionado. Seleccionamos nuestro ejercito.
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_QUEUED])
