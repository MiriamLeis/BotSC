import math
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from qtable import QTable

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_MOVE_UP = 0
_MOVE_DOWN = 1
_MOVE_RIGHT = 2
_MOVE_LEFT = 3
_MOVE_UP_RIGHT = 4
_MOVE_UP_LEFT = 5
_MOVE_DOWN_RIGHT = 6
_MOVE_DOWN_LEFT = 7

_SELECT_ALL = [0]
_NOT_QUEUED = [0]
_QUEUED = [1]

_MOVE_VAL = 3.5

possible_actions = [
    _MOVE_UP,
    _MOVE_DOWN,
    _MOVE_RIGHT,
    _MOVE_LEFT,
    _MOVE_UP_RIGHT,
    _MOVE_UP_LEFT,
    _MOVE_DOWN_RIGHT,
    _MOVE_DOWN_LEFT
]

class MoveToBeaconAgent(base_agent.BaseAgent):
    def __init__(self, episodes,load_qt=None, load_st=None):
        super(MoveToBeaconAgent, self).__init__()

        self.qtable = QTable(possible_actions, episodes,load_qt=load_qt, load_st=load_st)
        
    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)
        marineNextPosition = [0,0]
        # si podemos mover nuestro ejercito (si podemos mover algo)
        if _MOVE_SCREEN in obs.observation['available_actions']:
            state, dist, marinePos = get_state(obs)
            action = self.qtable.choose_action(state)
            func = actions.FunctionCall(_NO_OP, [])

            if  possible_actions[action] == _MOVE_UP:
                if(marinePos[1] - _MOVE_VAL < 3.5):
                    marinePos[1] +=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0], marinePos[1]- _MOVE_VAL]])
                marineNextPosition = [marinePos[0], marinePos[1]- _MOVE_VAL]

            elif possible_actions[action] == _MOVE_DOWN:
                if(marinePos[1] + _MOVE_VAL > 44.5):
                    marinePos[1] -=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0], marinePos[1] + _MOVE_VAL]])
                marineNextPosition = [marinePos[0], marinePos[1] + _MOVE_VAL]

            elif possible_actions[action] == _MOVE_RIGHT:
                if(marinePos[0] + _MOVE_VAL > 60.5):
                    marinePos[0] -=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0]+_MOVE_VAL, marinePos[1]]])
                marineNextPosition = [marinePos[0]+_MOVE_VAL, marinePos[1]]

            elif possible_actions[action] == _MOVE_LEFT:
                if(marinePos[0] - _MOVE_VAL < 3.5):
                    marinePos[0] +=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0]-_MOVE_VAL, marinePos[1]]])
                marineNextPosition = [marinePos[0]-_MOVE_VAL, marinePos[1]]

            elif possible_actions[action] == _MOVE_UP_RIGHT:
                if(marinePos[0] + _MOVE_VAL > 60.5):
                    marinePos[0] -=_MOVE_VAL
                if(marinePos[1] - _MOVE_VAL < 3.5):
                    marinePos[1] +=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0]+_MOVE_VAL, marinePos[1] - _MOVE_VAL]])
                marineNextPosition = [marinePos[0]+_MOVE_VAL, marinePos[1]- _MOVE_VAL]
            
            elif possible_actions[action] == _MOVE_UP_LEFT:
                if(marinePos[0]  - _MOVE_VAL < 3.5):
                    marinePos[0] +=_MOVE_VAL
                if(marinePos[1] - _MOVE_VAL < 3.5):
                    marinePos[1] +=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0]-_MOVE_VAL, marinePos[1] - _MOVE_VAL]])
                marineNextPosition = [marinePos[0]-_MOVE_VAL, marinePos[1]- _MOVE_VAL]
            
            elif possible_actions[action] == _MOVE_DOWN_RIGHT:
                if(marinePos[0] + _MOVE_VAL > 60.5):
                    marinePos[0] -=_MOVE_VAL
                if(marinePos[1] + _MOVE_VAL > 44.5):
                    marinePos[1] -=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0]+_MOVE_VAL, marinePos[1] + _MOVE_VAL]])
                marineNextPosition = [marinePos[0]+_MOVE_VAL, marinePos[1]+ _MOVE_VAL]

            else:
                if(marinePos[0]  - _MOVE_VAL < 3.5):
                    marinePos[0] +=_MOVE_VAL
                if(marinePos[1] + _MOVE_VAL > 44.5):
                    marinePos[1] -=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0]- _MOVE_VAL, marinePos[1] + _MOVE_VAL]])
                marineNextPosition = [marinePos[0] - _MOVE_VAL, marinePos[1] + _MOVE_VAL]
 
        # si no podemos movernos es porque no tenemos nada seleccionado. Seleccionamos nuestro ejercito.
        else:
            state = -1
            action = -1
            dist = -1
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            
        return state, action, func, dist, marineNextPosition


####################################################################

# devuelve el angulo de dos lineas
def ang(lineA, lineB):
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

# devuelve la posicion del marine
def get_marine_pos(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    marineys, marinexs = (ai_view == _PLAYER_SELF).nonzero()
    if len(marinexs) == 0:
        marinexs = np.array([0])
    if len(marineys) == 0:
        marineys = np.array([0])
    marinex, mariney = marinexs.mean(), marineys.mean()
    return marinex, mariney

# devuelve la posicion del beacon
def get_beacon_pos(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    beaconys, beaconxs = (ai_view == _PLAYER_NEUTRAL).nonzero()
    if len(beaconxs) == 0:
        beaconxs = np.array([0])
    if len(beaconys) == 0:
        beaconys = np.array([0])
    beaconx, beacony = beaconxs.mean(), beaconys.mean()
    return [beaconx, beacony]

# define el estado
def get_state(obs):
    # coge las poscion del marine y del beacon

    marinex, mariney = get_marine_pos(obs)
    beaconx, beacony = get_beacon_pos(obs)


    direction = [beaconx-marinex, beacony - mariney]
    dist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))
    
    vector_1 = [0, -1]

    np.linalg.norm(direction)

    angleD = ang(vector_1, direction)

    if direction[0] > 0:
        angleD = 360 - angleD

    state = -1
    if angleD >= 0 and angleD < 22.5 or angleD >= 337.5 and angleD < 360:
        state = 0
    elif angleD >= 22.5 and angleD < 67.5:
        state = 1
    elif angleD >= 67.5 and angleD < 112.5:
        state = 2
    elif angleD >= 112.5 and angleD < 157.5:
        state = 3
    elif angleD >= 157.5 and angleD < 202.5:
        state = 4
    elif angleD >= 202.5 and angleD < 247.5:
        state = 5
    elif angleD >= 247.5 and angleD < 292.5:
        state = 6
    elif angleD >= 292.5 and angleD < 337.5:
        state = 7

    return state, dist, [marinex, mariney]