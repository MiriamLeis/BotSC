from pysc2.env import sc2_env
from pysc2 import maps
from pysc2.lib import actions

from tqdm import tqdm

import numpy as np
import tensorflow as tf
import os
import time
import random

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import deepQ.agents.defeatzealots_2vs2 as class_agent #change path as needed

# Environment settings
EPISODES = 100
STEPS = 1_900


def main():
    # Create environment
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=16),
            use_feature_units=True)

    with sc2_env.SC2Env(map_name=maps.get(class_agent.MAP_NAME),
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        visualize=False,
                        agent_interface_format=AGENT_INTERFACE_FORMAT,
                        step_mul= 1) as env:

        agent = class_agent.Agent(True)

        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
            print()

            obs = env.reset()
            step = 1
            
            func, action = agent.prepare(obs[0], episode - 1)
            current_state = agent.get_state(obs[0])

            obs = env.step(actions=[func])
        
            done = False
            end = False

            actualTime = 2.0
            timeForAction = 0.2
            lastTime = ((obs[0]).observation["game_loop"] / 16)

            for s in range(STEPS):
                end = agent.get_end(obs[0])
                if end:
                    break

                # get deltaTime
                realTime = ((obs[0]).observation["game_loop"] / 16)
                delta = realTime - lastTime
                lastTime = realTime

                if actualTime >= timeForAction:
                    current_state = agent.get_state(obs[0])

                    agent.update(obs[0], delta)

                    action = agent.get_max_action(current_state)

                    func = agent.get_action(obs[0], action)
                    actualTime = 0
                else:
                    actualTime += delta

                func = agent.check_action_available(obs[0], action, func)

                obs, end = agent.step(env, func)
                if end:
                    break
                    
                step += 1
            

main()