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
flags.DEFINE_boolean("start_from_scratch", True, "If you want to continue with an existing Q-table")
flags.DEFINE_integer("max_episodes", 35, "Number of episodes")

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_MOVE_TO_BEACON = 0
_MOVE_RAND = 1
_MOVE_MIDDLE = 2

_SELECT_ALL = [0]
_NOT_QUEUED = [0]

possible_actions = [
    _MOVE_TO_BEACON,
    _MOVE_RAND,
    _MOVE_MIDDLE
]

# define el estado
def get_state(obs):
    # coge las poscion del marine y del beacon
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    beaconxs, beaconys = (ai_view == _PLAYER_NEUTRAL).nonzero()
    marinexs, marineys = (ai_view == _PLAYER_SELF).nonzero()
    marinex, mariney = marinexs.mean(), marineys.mean()
        
    marine_on_beacon = np.min(beaconxs) <= marinex <=  np.max(beaconxs) and np.min(beaconys) <= mariney <=  np.max(beaconys)

    return int(marine_on_beacon), [beaconxs, beaconys]

class QTable(object):
    def __init__(self, actions, load_qt=None, load_st=None):
        self.actions = actions
        self.load_qt = load_qt

        self.states_list = self.load_states(load_st)
        set(self.states_list)
        
        self.q_table = self.load_qtable(load_qt)
        
    def choose_action(self, state):
            
        idx = list(self.states_list).index(state)
        q_values = self.q_table[idx]
        return int(np.argmax(q_values))
    
    def load_qtable(self, filepath):
        return np.load(filepath)
        
    def load_states(self, filepath):
        return np.load(filepath)
    
class MoveToBeaconAgent(base_agent.BaseAgent):
    def __init__(self, load_qt=None, load_st=None):
        super(MoveToBeaconAgent, self).__init__()

        self.qtable = QTable(possible_actions, load_qt=load_qt, load_st=load_st)
        
    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)

        # si podemos mover nuestro ejercito (si podemos mover algo)
        if _MOVE_SCREEN in obs.observation['available_actions']:
            state, beacon_pos = get_state(obs)
            action = self.qtable.choose_action(state)
            func = actions.FunctionCall(_NO_OP, [])
            
            if  possible_actions[action] == _MOVE_TO_BEACON:
                beacon_x, beacon_y = beacon_pos[0].mean(), beacon_pos[1].mean()
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_y, beacon_x]])

            elif possible_actions[action] == _MOVE_RAND:
                beacon_x, beacon_y = beacon_pos[0].max(), beacon_pos[1].max()
                movex, movey = np.random.randint(beacon_x, 63), np.random.randint(beacon_y, 63)
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [movey, movex]])

            elif possible_actions[action] == _MOVE_MIDDLE:
                func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [32, 32]])
 
        # si no podemos movernos es porque no tenemos nada seleccionado. Seleccionamos nuestro ejercito.
        else:
            state = -1
            action = -1
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            
        return state, func

def main():
    FLAGS(sys.argv)
    MAX_STEPS = 400

    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=16, minimap=16))

    with sc2_env.SC2Env(map_name=maps.get('MoveToBeacon'),
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        visualize=False,
                        agent_interface_format=AGENT_INTERFACE_FORMAT) as env:
    
        agent = MoveToBeaconAgent(load_qt='moveToBeaconAgent_qtable.npy', load_st='moveToBeaconAgent_states.npy')

        for i in range(FLAGS.max_episodes):
            num_beacons = 0
            print('Starting episode {}'.format(i))
            obs = env.reset()
            for j in range(MAX_STEPS):
                state, func = agent.step(obs[0])
                obs = env.step(actions=[func])
                if state == 1:
                    num_beacons += 1

            print('Episode Beacons: {}'.format(num_beacons))
            
        if FLAGS.save_replay:
            env.save_replay(MoveToBeaconAgent.__name__)

if __name__ == "__main__":
    main()