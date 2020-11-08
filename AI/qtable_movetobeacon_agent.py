import math

import numpy as np
import pandas as pd

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.agents import base_agent

# constantes para las caracteristicas
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 0  # jugador
_PLAYER_NEUTRAL = 3  # beacon

# constantes para las acciones
_QUEUED = [0]
_NOT_QUEUED = [0]

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

ACTION_MOVE_NORTH = 'movenorth'
ACTION_MOVE_EAST = 'moveeast'
ACTION_MOVE_SOUTH = 'movesouth'
ACTION_MOVE_WEST = 'movewest'

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
MOVE_VALUE = 1.0
WORLD_WIDTH = 64
WORLD_HEIGHT = 64


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(state)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.iloc[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()

        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        self.check_state_exist(state)

        q_predict = self.q_table.iloc[state, action]

        # almacena la recompensa por:
        # la accion que ha realizado + lo bueno que es el estado al que te ha llevado
        q_target = reward + self.gamma * self.q_table.iloc[state_, :].max()

        # actualizar la recompensa de ese estado con esa accion dependiendo de lo que hubiese antes
        self.q_table.iloc[state, action] = (1 - self.lr) * q_predict + self.lr * (q_target)
        # self.q_table.iloc[state, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.colums, name=state))


def get_player_location(obs):
    ai_view = obs.observation['screen'][_PLAYER_RELATIVE]
    return (ai_view == _PLAYER_SELF).nonzero()


def get_beacon_location(obs):
    ai_view = obs.observation['screen'][_PLAYER_RELATIVE]
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

            player_position = [player_xs.mean(), player_ys.mean()]
            beacon_position = [beacon_xs.mean(), beacon_ys.mean()]

            # creamos el estado actual
            current_state = [
                player_position,
                beacon_position
            ]

            # dar recompensas a la accion anterior
            if self.previous_action is not None:
                reward = 0

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
            if actual_action == ACTION_MOVE_NORTH:
                target = [player_position[0], player_position[1] + MOVE_VALUE]
                if target[1] >= WORLD_HEIGHT:
                    target[1] = WORLD_HEIGHT - 1

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            elif actual_action == ACTION_MOVE_SOUTH:
                target = [player_position[0], player_position[1] - MOVE_VALUE]
                if target[1] < 0:
                    target[1] = 0

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            elif actual_action == ACTION_MOVE_EAST:
                target = [player_position[0] + MOVE_VALUE, player_position[1]]
                if target[1] >= WORLD_WIDTH:
                    target[1] = WORLD_WIDTH - 1

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            elif actual_action == ACTION_MOVE_WEST:
                target = [player_position[0], player_position[1] + MOVE_VALUE]
                if target[1] < 0:
                    target[1] = 0

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            return actions.FunctionCall(_NO_OP, [])

        # si no podemos movernos es porque no tenemos nada seleccionado. Seleccionamos nuestro ejercito.
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_QUEUED])
