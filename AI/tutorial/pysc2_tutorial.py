from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import Adam
import keras
import tensorflow as tf

"""
    Use GPU
"""
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)

from collections import deque
from tqdm import tqdm
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env
from pysc2 import maps
import sys
import numpy as np

from absl import flags
import time
import random
import math
import time

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

 # python ignor
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 50  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '1_64_4'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 30
STEPS = 1_900

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


###############################


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_SELECT_ALL = [0]
_NOT_QUEUED = [0]
_QUEUED = [1]

MOVE_VAL = 3.5

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

possible_actions = [
    UP,
    DOWN,
    RIGHT,
    LEFT
]

###############################

class DQNAgent:
    def __init__(self):
        #actions
        self.num_actions = 4

        # main model
        # model that we are not fitting every step
        # gets trained every step
        self.model = self.create_model()

        # target model
        # this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights()) # do it again after a while

        # deque -> array or list 
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # track internally when we are ready to update target_model
        self.target_update_counter = 0

    def create_model(self):
        # layers
        inputs = Input(shape=(8,))
        x = Dense(25, activation='relu')(inputs)
        outputs = Dense(self.num_actions, activation='softmax')(x)

        # creation
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
    
        model.summary()
        print(model.get_weights())
        return model

    def update_replay_memory(self, transition):
        # transition -> our observation space
        self.replay_memory.append(transition)

    def get_qs(self, state):
        # return last layer of neural network
        stateArray = np.array(state)
        return self.model.predict(stateArray.reshape(-1, *stateArray.shape))[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # / 255 cuz we want to normalize in this case... but this is just for images.
        # so this is cuz we want values from 0 to 1
        current_states = np.array([transition[0] for transition in self.replay_memory])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in self.replay_memory])
        future_qs_list = self.target_model.predict(new_current_states)

        X = [] # feature sets   /  images
        y = [] # labels         /  actions

        # current_states get from index 0 so current_state has to be in 0 position
        # same with new_current_state
        ## ALGORITHM
        for index, (current_state, action, reward, new_current_state, done) in enumerate(self.replay_memory):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        # we do fit only if terminal_state, otherwise we fit None
        self.model.fit(np.array(X), np.array(y), batch_size=MIN_REPLAY_MEMORY_SIZE, verbose=2, 
            shuffle=False)

        self.replay_memory.clear()

        #updating to determinate if we want to update target_model yet
        
        self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



###############################

# return marine position
def get_marine_pos(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    marineys, marinexs = (ai_view == _PLAYER_SELF).nonzero()
    if len(marinexs) == 0:
        marinexs = np.array([0])
    if len(marineys) == 0:
        marineys = np.array([0])
    marinex, mariney = marinexs.mean(), marineys.mean()
    return marinex, mariney

# return beacon position
def get_beacon_pos(obs):
    ai_view = obs.observation['feature_screen'][_PLAYER_RELATIVE]
    beaconys, beaconxs = (ai_view == _PLAYER_NEUTRAL).nonzero()
    if len(beaconxs) == 0:
        beaconxs = np.array([0])
    if len(beaconys) == 0:
        beaconys = np.array([0])
    beaconx, beacony = beaconxs.mean(), beaconys.mean()
    return [beaconx, beacony]

# angle formed from two lines 
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

def get_state(obs):
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
        state = [1,0,0,0,0,0,0,0]
    elif angleD >= 22.5 and angleD < 67.5:
        state = [0,1,0,0,0,0,0,0]
    elif angleD >= 67.5 and angleD < 112.5:
        state = [0,0,1,0,0,0,0,0]
    elif angleD >= 112.5 and angleD < 157.5:
        state = [0,0,0,1,0,0,0,0]
    elif angleD >= 157.5 and angleD < 202.5:
        state = [0,0,0,0,1,0,0,0]
    elif angleD >= 202.5 and angleD < 247.5:
        state = [0,0,0,0,0,1,0,0]
    elif angleD >= 247.5 and angleD < 292.5:
        state = [0,0,0,0,0,0,1,0]
    elif angleD >= 292.5 and angleD < 337.5:
        state = [0,0,0,0,0,0,0,1]

    return state

def check_done(obs, beacon_actual_pos, last_step):
    beacon_new_pos = get_beacon_pos(obs)
    # if we get beacon or it's the last step
    if beacon_actual_pos[0] != beacon_new_pos[0] or beacon_actual_pos[1] != beacon_new_pos[1] or last_step:
        return True, [beacon_new_pos[0], beacon_new_pos[1]]
    return False, [beacon_actual_pos[0], beacon_actual_pos[1]]

def get_action(obs, action):
    marinex, mariney = get_marine_pos(obs)
    func = actions.FunctionCall(_NO_OP, [])
    
    if  possible_actions[action] == UP:
        if(mariney - MOVE_VAL < 3.5):
            mariney += MOVE_VAL
        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex, mariney - MOVE_VAL]])
        marineNextPosition = [marinex, mariney - MOVE_VAL]

    elif possible_actions[action] == DOWN:
        if(mariney + MOVE_VAL > 44.5):
            mariney -= MOVE_VAL

        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex, mariney + MOVE_VAL]])
        marineNextPosition = [marinex, mariney + MOVE_VAL]

    elif possible_actions[action] == RIGHT:
        if(marinex + MOVE_VAL > 60.5):
            marinex -= MOVE_VAL
        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex + MOVE_VAL, mariney]])
        marineNextPosition = [marinex + MOVE_VAL, mariney]

    else:
        if(marinex - MOVE_VAL < 3.5):
            marinex += MOVE_VAL

        func = actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marinex - MOVE_VAL, mariney]])
        marineNextPosition = [marinex - MOVE_VAL, mariney]

    return func

def get_reward(obs, oldDist):
    return oldDist - get_dist(obs)

def get_dist(obs):
    marinex, mariney = get_marine_pos(obs)
    beaconx, beacony = get_beacon_pos(obs)

    newDist = math.sqrt(pow(marinex - beaconx, 2) + pow(mariney - beacony, 2))
    return newDist

def main():
    # Create environment
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=16))

    with sc2_env.SC2Env(map_name=maps.get('MoveToBeacon'),
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        visualize=False,
                        agent_interface_format=AGENT_INTERFACE_FORMAT,
                        step_mul= 1) as env:

        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        print(sess)

        #tf.compat.v1.keras.backend._get_available_gpus()
        agent = DQNAgent()

        epsilon = 1
        ep_rewards = [-200]

        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        ep = 0
        for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
            # decay epsilon
            epsilon = 1 - (ep/(EPISODES - 10))
            print(epsilon)

            obs = env.reset()
            step = 1

            beacon_actual_pos = get_beacon_pos(obs[0])
            oldDist = get_dist(obs[0])

            current_state = get_state(obs[0])

            # select marine
            func = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            obs = env.step(actions=[func])

            action = 0

            done = False

            actualTime = 2.0
            timeForAction = 0.75
            lastTime = ((obs[0]).observation["game_loop"] / 16)
        
            ep += 1

            for s in range(STEPS):
                # get deltaTime
                realTime = ((obs[0]).observation["game_loop"] / 16)
                delta = realTime - lastTime
                lastTime = realTime
                
                if actualTime >= timeForAction:
                   # get new state
                    new_state = get_state(obs[0])

                    done, beacon_actual_pos = check_done(obs[0], beacon_actual_pos, s==STEPS-1)

                    # get reward of our action
                    reward = get_reward(obs[0], oldDist)
                    if reward < 0:
                        reward = 0
                    if reward > 1:
                        reward = 1

                    oldDist = get_dist(obs[0])

                    # Every step we update replay memory and train main network
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
                    agent.train(done, step)

                    current_state = new_state

                    if np.random.random() > epsilon:
                        # choose action

                        casos = agent.get_qs(current_state)
                        action = np.argmax(casos)
                        print("Estado : ", current_state)
                        print("Acciones : ", casos)
                    else:
                        # get random action
                        action = np.random.randint(0, agent.num_actions)

                    func = get_action(obs[0], action)
                    actualTime = 0
                else:
                    actualTime += delta

                obs = env.step(actions=[func])
                
                step += 1
            

main()