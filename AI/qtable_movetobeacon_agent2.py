
import math
import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags

FLAGS = flags.FLAGS
FLAGS(['run_sc2'])
_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_SELECT_ALL = [0]
_NOT_QUEUED = [0]
EPS_START = 0.9
EPS_END = 0.025
EPS_DECAY = 2500
# define our actions
# it can choose to move to
# the beacon or to do nothing
# it can select the marine or deselect
# the marine, it can move to a random point
possible_actions = [
    _NO_OP,
    _SELECT_ARMY,
    _SELECT_POINT,
    _MOVE_SCREEN,
    _MOVE_RAND,
    _MOVE_MIDDLE
]
def get_eps_threshold(steps_done):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

# define the state
def get_state(obs):
    # get the positions of the marine and the beacon
    ai_view = obs.observation['feature_screen'][_AI_RELATIVE]
    beaconxs, beaconys = (ai_view == _AI_NEUTRAL).nonzero()
    marinexs, marineys = (ai_view == _AI_SELF).nonzero()
    marinex, mariney = marinexs.mean(), marineys.mean()
        
    marine_on_beacon = np.min(beaconxs) <= marinex <=  np.max(beaconxs) and np.min(beaconys) <= mariney <=  np.max(beaconys)
        
    # get a 1 or 0 for whether or not our marine is selected
    ai_selected = obs.observation['feature_screen'][_AI_SELECTED]
    marine_selected = int((ai_selected == 1).any())
    
    return (marine_selected, int(marine_on_beacon)), [beaconxs, beaconys]

class QTable(object):
    def __init__(self, actions, lr=0.01, reward_decay=0.9, load_qt=None, load_st=None):
        self.lr = lr
        self.actions = actions
        self.reward_decay = reward_decay
        self.states_list = set()
        self.load_qt = load_qt
        if load_st:
            temp = self.load_states(load_st)
            self.states_list = set([tuple(temp[i]) for i in range(len(temp))])
        
        if load_qt:
            self.q_table = self.load_qtable(load_qt)
        else:
            self.q_table = np.zeros((0, len(possible_actions))) # create a Q table
        
    def get_action(self, state):
        if not self.load_qt and np.random.rand() < get_eps_threshold(steps):
            return np.random.randint(0, len(self.actions))
        else:
            if state not in self.states_list:
                self.add_state(state)
            idx = list(self.states_list).index(state)
            q_values = self.q_table[idx]
            return int(np.argmax(q_values))
    
    def add_state(self, state):
        self.q_table = np.vstack([self.q_table, np.zeros((1, len(possible_actions)))])
        self.states_list.add(state)
    
    def update_qtable(self, state, next_state, action, reward):
        if state not in self.states_list:
            self.add_state(state)
        if next_state not in self.states_list:
            self.add_state(next_state)
        # how much reward 
        state_idx = list(self.states_list).index(state)
        next_state_idx = list(self.states_list).index(next_state)
        # calculate q labels
        q_state = self.q_table[state_idx, action]
        q_next_state = self.q_table[next_state_idx].max()
        q_targets = reward + (self.reward_decay * q_next_state)
        # calculate our loss 
        loss = q_targets - q_state
        # update the q value for this state/action pair
        self.q_table[state_idx, action] += self.lr * loss
        return loss
    
    def get_size(self):
        print(self.q_table.shape)
        
    def save_qtable(self, filepath):
        np.save(filepath, self.q_table)
        
    def load_qtable(self, filepath):
        return np.load(filepath)
        
    def save_states(self, filepath):
        temp = np.array(list(self.states_list))
        np.save(filepath, temp)
        
    def load_states(self, filepath):
        return np.load(filepath)
    
class Agent3(base_agent.BaseAgent):
    def __init__(self, load_qt=None, load_st=None):
        super(Agent3, self).__init__()
        self.qtable = QTable(possible_actions, load_qt=load_qt, load_st=load_st)
        
    def step(self, obs):
        '''Step function gets called automatically by pysc2 environment'''
        super(Agent3, self).step(obs)
        state, beacon_pos = get_state(obs)
        action = self.qtable.get_action(state)
        func = actions.FunctionCall(_NO_OP, [])
        
        if possible_actions[action] == _NO_OP:
            func = actions.FunctionCall(_NO_OP, [])
        elif state[0] and possible_actions[action] == _MOVE_SCREEN:
            beacon_x, beacon_y = beacon_pos[0].mean(), beacon_pos[1].mean()
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_y, beacon_x]])
        elif possible_actions[action] == _SELECT_ARMY:
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        elif state[0] and possible_actions[action] == _SELECT_POINT:
            ai_view = obs.observation['feature_screen'][_AI_RELATIVE]
            backgroundxs, backgroundys = (ai_view == _BACKGROUND).nonzero()
            point = np.random.randint(0, len(backgroundxs))
            backgroundx, backgroundy = backgroundxs[point], backgroundys[point]
            func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [backgroundy, backgroundx]])
        elif state[0] and possible_actions[action] == _MOVE_RAND:
            # move somewhere that is not the beacon
            beacon_x, beacon_y = beacon_pos[0].max(), beacon_pos[1].max()
            movex, movey = np.random.randint(beacon_x, 63), np.random.randint(beacon_y, 63)
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [movey, movex]])
        elif state[0] and possible_actions[action] == _MOVE_MIDDLE:
            func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [32, 32]])
        return state, action, func

viz = False
save_replay = False
steps_per_episode = 0 # 0 actually means unlimited
MAX_EPISODES =35
MAX_STEPS = 400
steps = 0

# create a map
beacon_map = maps.get('MoveToBeacon')
AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(screen=16, minimap=16))

# create an envirnoment
with sc2_env.SC2Env(map_name=beacon_map,
                    players=[sc2_env.Agent(sc2_env.Race.terran)],
                    visualize=viz,
                    agent_interface_format=AGENT_INTERFACE_FORMAT) as env:
    agent = Agent3('agent3_qtable.npy','agent3_states.npy')
    for i in range(MAX_EPISODES):
        print('Starting episode {}'.format(i))
        ep_reward = 0
        obs = env.reset()
        for j in range(MAX_STEPS):
            steps += 1
            state, action, func = agent.step(obs[0])
            obs = env.step(actions=[func])
            next_state, _ = get_state(obs[0])
            reward = obs[0].reward
            ep_reward += reward
            loss = agent.qtable.update_qtable(state, next_state, action, reward)
        print('Episode Reward: {}, Explore threshold: {}, Q loss: {}'.format(ep_reward, get_eps_threshold(steps), loss))
    if save_replay:
        env.save_replay(Agent3.__name__)
    agent.qtable.save_qtable('agent3_qtable.npy')
    agent.qtable.save_states('agent3_states.npy')