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

import agents.defeatzealots1Reward as class_agent #change path as needed

# Environment settings
EPISODES = 10
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

        epsilon = 1
        ep_rewards = [-200]

        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        ep = 0
        end = False
        action = -1
        current_state = -1
        for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"):
            print()

            # decay epsilon
            epsilon = 0

            obs = env.reset()
            step = 1
            
            if end:
                # get and learn reward for last action
                reward = agent.get_reward(obs[0], action)
                agent.update_replay_memory((current_state, action, reward, agent.get_state(obs[0]), agent.check_done(obs[0], STEPS-1)))

            # prepare new step
            func, action = agent.prepare(obs[0])
            current_state = agent.get_state(obs[0])

            obs = env.step(actions=[func])

            done = False
            end = False

            actualTime = 5.0
            timeForAction = 0.5
            lastTime = ((obs[0]).observation["game_loop"] / 16)
        
            ep += 1

            for s in range(STEPS):
                # leave episode if we ended
                end = agent.get_end(obs[0])
                if end:
                    break

                # get deltaTime
                realTime = ((obs[0]).observation["game_loop"] / 16)
                delta = realTime - lastTime
                lastTime = realTime

                if actualTime >= timeForAction:
                   # get new state
                    new_state = agent.get_state(obs[0])

                    done = agent.check_done(obs[0], step == STEPS-1)

                    # get reward of our action
                    reward = agent.get_reward(obs[0], action)

                    print("Recompensa : ", reward)
                    print("Accion : ", action)
                    print("Estado : ", current_state)
                    print("Nuevo Estado: ", new_state)
                    print("---")

                    agent.update(obs[0], delta)

                    # Every step we update replay memory and train main network
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
                    agent.train(step)

                    current_state = new_state


                    if np.random.random() > epsilon:
                        # choose action
                        casos = agent.get_qs(current_state)
                        action = np.argmax(casos)
                        print("Estado : ", current_state)
                        print("Acciones : ", casos)
                        
                    else:
                        # get random action
                        action = np.random.randint(0, agent.get_num_actions())

                    func = agent.get_action(obs[0], action)
                    actualTime = 0

                else:
                    actualTime += delta

                func = agent.check_action_available(obs[0], action, func)
                obs = env.step(actions=[func])
                
                step += 1

        agent.saveModel(os.getcwd() + '/models/' + class_agent.FILE_NAME + '.h5')
            

main()