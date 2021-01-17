from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env
from pysc2 import maps

import numpy as np
import tensorflow as tf
import time
import random
import math

 # python ignore _
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '1_256_64_4'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000
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


###############################

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


###############################

class DQNAgent:
    def __init__(self):
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
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        #actions
        self.num_actions = 4

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(1,)))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64))

        model.add(Dense(self.num_actions, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # transition -> our observation space
        self.replay_memory.append(transition)

    def get_qs(self, state):
        # return last layer of neural network
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # / 255 cuz we want to normalize in this case... but this is just for images.
        # so this is cuz we want values from 0 to 1
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = [] # feature sets   /  images
        y = [] # labels         /  actions

        # current_states get from index 0 so current_state has to be in 0 position
        # same with new_current_state
        ## ALGORITHM
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
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
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, 
            shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        #updating to determinate if we want to update target_model yet
        if terminal_state: 
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

    return state

def check_done(obs, beacon_actual_pos, last_step):
    beacon_new_pos = get_beacon_pos(obs[0])
    # if we get beacon or it's the last step
    if beacon_actual_pos[0] != beacon_new_pos[0] or beacon_actual_pos[1] != beacon_new_pos[1] or last_step:
        return True, [beacon_new_pos[0], beacon_new_pos[1]]
    return False, [beacon_actual_pos[0], beacon_actual_pos[1]]

def main():
    # Create environment
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=16))

    with sc2_env.SC2Env(map_name=maps.get('MoveToBeacon'),
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        visualize=False,
                        agent_interface_format=AGENT_INTERFACE_FORMAT,
                        step_mul= 1) as env:

        agent = DQNAgent()

        epsilon = 1
        ep_rewards = [-200]

        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
            obs = env.reset()

            agent.tensorboard.step = episode

            episode_reward = 0
            step = 1

            beacon_actual_pos = get_beacon_pos(obs[0])
            current_state = get_state(obs)

            # select marine
            action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            obs = env.step(actions=[action])

            done = False

            for s in range(STEPS):
                if np.random.random() > epsilon:
                    # choose action
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # get random action
                    action = np.random.randint(0, agent.num_actions)

                obs = env.step(actions=[action])

                # get new state
                new_state = get_state(obs)

                done, beacon_actual_pos = check_done(obs, beacon_actual_pos, s==STEPS-1)

                # get reward of our action
                reward = obs[0].reward
                episode_reward += reward

                # Every step we update replay memory and train main network
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done, step)

                current_state = new_state
                step += 1

                if done:
                    # Append episode reward to a list and log stats (every given number of episodes)
                    ep_rewards.append(episode_reward)
                    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

                        # Save model, but only when min reward is greater or equal a set value
                        if min_reward >= MIN_REWARD:
                            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

                    # Decay epsilon
                    if epsilon > MIN_EPSILON:
                        epsilon *= EPSILON_DECAY
                        epsilon = max(MIN_EPSILON, epsilon)
            

main()