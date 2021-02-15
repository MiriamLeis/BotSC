from pysc2.env import sc2_env
from pysc2 import maps

from tqdm import tqdm

import numpy as np
import tensorflow as tf
import time
import random

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import dq_network
import agents.movetobeacon as class_agent #change path as needed

# Environment settings
EPISODES = 30
STEPS = 1_900


def main():
    # Create environment
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=16))

    with sc2_env.SC2Env(map_name=maps.get(class_agent.MAP_NAME),
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        visualize=False,
                        agent_interface_format=AGENT_INTERFACE_FORMAT,
                        step_mul= 1) as env:

        agent = class_agent.Agent()

        dq_agent = dq_network.DQNAgent(num_actions=agent.get_num_actions(),
                                        num_states=agent.get_num_states(),
                                        discount=class_agent.DISCOUNT,
                                        rep_mem_size=class_agent.REPLAY_MEMORY_SIZE,
                                        min_rep_mem_size=class_agent.MIN_REPLAY_MEMORY_SIZE,
                                        update_time=class_agent.UPDATE_TARGET_EVERY)

        epsilon = 1
        ep_rewards = [-200]

        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        ep = 0
        for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
            # decay epsilon
            epsilon = 1 - (ep/(EPISODES - 10))

            obs = env.reset()
            step = 1
            
            func, action = agent.prepare(obs[0])
            current_state = agent.get_state(obs[0])

            obs = env.step(actions=[func])

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
                    new_state = agent.get_state(obs[0])

                    done = agent.check_done(obs[0], STEPS-1)

                    # get reward of our action
                    reward = agent.get_reward(obs[0])
                    if reward < 0:
                        reward = 0
                    if reward > 1:
                        reward = 1

                    agent.update(obs[0])

                    # Every step we update replay memory and train main network
                    dq_agent.update_replay_memory((current_state, action, reward, new_state, done))
                    dq_agent.train(done, step)

                    current_state = new_state

                    if np.random.random() > epsilon:
                        # choose action
                        casos = dq_agent.get_qs(current_state)
                        action = np.argmax(casos)
                    else:
                        # get random action
                        action = np.random.randint(0, dq_agent.num_actions)

                    func = agent.get_action(obs[0], action)
                    actualTime = 0

                else:
                    actualTime += delta

                obs = env.step(actions=[func])
                
                step += 1
            

main()