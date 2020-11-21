import math
import numpy as np
import sys
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("save_replay", False, "If you want to save replay")
flags.DEFINE_integer("max_episodes", 5, "Number of episodes")

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

_SELECT_ALL = [0]
_NOT_QUEUED = [0]
_QUEUED = [1]

possible_actions = [
    _MOVE_UP,
    _MOVE_DOWN,
    _MOVE_RIGHT,
    _MOVE_LEFT
]

_MOVE_VAL = 3.5

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

def get_marine_pos(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    marineys, marinexs = (ai_view == _PLAYER_SELF).nonzero()
    if len(marinexs) == 0:
        marinexs = np.array([0])
    if len(marineys) == 0:
        marineys = np.array([0])
    marinex, mariney = marinexs.mean(), marineys.mean()

    return marinex, mariney

def get_beacon_pos(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    beaconys, beaconxs = (ai_view == _PLAYER_NEUTRAL).nonzero()


    beaconx, beacony = beaconxs.mean(), beaconys.mean()
    return beaconx, beacony

# define el estado
def get_state(obs):
    # coge las poscion del marine y del beacon
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    beaconys, beaconxs = (ai_view == _PLAYER_NEUTRAL).nonzero()
    marineys, marinexs = (ai_view == _PLAYER_SELF).nonzero()


    if len(marinexs) == 0:
        marinexs = np.array([0])
    if len(marineys) == 0:
        marineys = np.array([0])

    marinex, mariney = marinexs.mean(), marineys.mean()
    beaconx, beacony = beaconxs.mean(), beaconys.mean()

    direction = [beaconx-marinex, beacony - mariney]
    dist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))
    
    vector_1 = [0, -1]

    np.linalg.norm(vector_1)
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

class QTable(object):
    def __init__(self, actions, lr=0.5, reward_decay=0.95, e_greedy=1.0,load_qt=None, load_st=None):
        self.lr = lr
        self.actions = actions
        self.epsilon = e_greedy
        self.gamma = reward_decay
        self.load_qt = load_qt
        if load_st:
            self.states_list = self.load_states(load_st)
            set(self.states_list)
        else:
            self.states_list = set()
        
        if load_qt:
            self.q_table = self.load_qtable(load_qt)
            print(self.q_table)
        else:
            self.q_table = np.zeros((0, len(possible_actions))) # crea la tabla Q
        
    def choose_action(self, state):
        self.check_state_exist(state)
            
        idx = list(self.states_list).index(state)
        q_values = self.q_table[idx]
        return int(np.argmax(q_values))


    def check_state_exist(self, state):
        if state not in self.states_list:
            self.q_table = np.vstack([self.q_table, np.zeros((1, len(possible_actions)))])
            self.states_list.add(state)
        
    def load_qtable(self, filepath):
        return np.load(filepath)     
        
    def load_states(self, filepath):
        return np.load(filepath)

    def print_QTable(self):
        print(self.q_table)
    
class MoveToBeaconAgent(base_agent.BaseAgent):
    def __init__(self, load_qt=None, load_st=None):
        super(MoveToBeaconAgent, self).__init__()

        self.qtable = QTable(possible_actions, load_qt=load_qt, load_st=load_st)
        
    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)
        marineNextPosition = [0,0]
        # si podemos mover nuestro ejercito (si podemos mover algo)
        if _MOVE_SCREEN in obs.observation['available_actions']:
            state, dist, marinePos = get_state(obs)
            action = self.qtable.choose_action(state)
            func = actions.FunctionCall(_NO_OP, [])

            beacon = get_beacon_pos(obs)

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
            else:
                if(marinePos[0] - _MOVE_VAL < 3.5):
                    marinePos[0] +=_MOVE_VAL
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinePos[0]-_MOVE_VAL, marinePos[1]]])
                marineNextPosition = [marinePos[0]-_MOVE_VAL, marinePos[1]]
 
        # si no podemos movernos es porque no tenemos nada seleccionado. Seleccionamos nuestro ejercito.
        else:
            state = -1
            action = -1
            dist = -1
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            
        return state, action, func, dist, marineNextPosition


def main():
    FLAGS(sys.argv)
    MAX_STEPS = 1900
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=16))

    with sc2_env.SC2Env(map_name=maps.get('MoveToBeacon'),
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        visualize=False,
                        agent_interface_format=AGENT_INTERFACE_FORMAT,
                        step_mul= 1) as env:
    

        agent = MoveToBeaconAgent(load_qt='moveToBeaconAgent_qtable_V2_2.npy', load_st='moveToBeaconAgent_states_V2_2.npy')
        agent.qtable.print_QTable()
        for i in range(FLAGS.max_episodes):
            print('Starting episode {}'.format(i))
            ep_reward = 0
            obs = env.reset()
            print(i, FLAGS.max_episodes)
            marineActualPos = get_marine_pos(obs[0])

            marineNextPosition = np.copy(marineActualPos)
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            state = -1
            oldDist = 0
            action = -1

            for j in range(MAX_STEPS):
                marineActualPos = get_marine_pos(obs[0])

                if (marineNextPosition[0] <= marineActualPos[0] + 0.5 and marineNextPosition[0] >= marineActualPos[0] - 0.5 and marineNextPosition[1] <= marineActualPos[1] + 0.5 and marineNextPosition[1] >= marineActualPos[1] - 0.5):

                    obs = env.step(actions=[func])

                    state, action, func, oldDist, marinePosibleNextPosition = agent.step(obs[0])
                    marineNextPosition = marinePosibleNextPosition
                    #print("Estado:", state)
                    #print("Accion:", action)
                    #print("Proxima Posicion:", marineNextPosition)
                    #print("Posicion Actual:",  get_marine_pos(obs[0]))
                    #print("Posicion Beacon:",  get_beacon_pos(obs[0]))


                elif _MOVE_SCREEN in obs[0].observation['available_actions']:
                    if marineActualPos[0] == 0.0 and marineActualPos[1] == 0.0:
                        print("error")
                        beacon = get_beacon_pos(obs[0])
                        marineNextPosition = [beacon[0], beacon[1]]     
                        obs = env.step(actions=[actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon[0], beacon[1]]])])
                    else:
                        marineNextPosition = [marineActualPos[0], marineActualPos[1]]    
                        obs = env.step(actions=[func])
                else:

                    obs = env.step(actions=[actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

            print('Episode Reward: {}'.format(ep_reward))
            
        if FLAGS.save_replay:
            env.save_replay(MoveToBeaconAgent.__name__)


if __name__ == "__main__":
    main()