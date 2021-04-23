import random
import sys
import os

from tqdm import tqdm

from pysc2_env import PySC2 as Environment # environment
from movetobeacon import MoveToBeacon as EnvAgent # environment agent
from qagent import QAgent as Agent # algorithm agent

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes', 1000, 'Number of episodes.', lower_bound=0)
flags.DEFINE_integer('steps', 1900, 'Steps from each episode.', lower_bound=0)
flags.DEFINE_integer('episodes_for_save', 10, 'Episodes until backup save.', lower_bound=0)
flags.DEFINE_float('time_for_action', 0.5, 'Time until choose new action.', lower_bound=0.0)
flags.DEFINE_boolean('load', False, 'Will load information.')
flags.DEFINE_string('filepath', os.getcwd() + '\\qlearning\\saves\\mtb', 'Filepath for load or save. Path must exist.')
flags.mark_flag_as_required('filepath')

def main():
    FLAGS(sys.argv)

    agent = Agent(agent=EnvAgent(),total_episodes=FLAGS.episodes)
    if FLAGS.load:
        agent.load(FLAGS.filepath)

    env = Environment(args=agent.get_info())

    random.seed(1)
    end = False

    for episode in tqdm(range(1,FLAGS.episodes+1), ascii=True, unit="episode"):
        obs = env.reset()

        if (episode % FLAGS.episodes_for_save) == 0:
            agent.save('saves_episode\\' + FLAGS.filepath + '\\' + str(episode))

        agent.prepare(env=obs[0],ep=episode-1)

        actualTime = sys.maxsize
        lastTime = 0.0

        for step in range(FLAGS.steps):
            # get deltaTime
            realTime = ((obs[0]).observation["game_loop"] / 16)
            deltaTime = realTime - lastTime
            lastTime = realTime

            if actualTime >= FLAGS.time_for_action:
                agent.update(env=obs[0], deltaTime=deltaTime)
                agent.train()
                agent.choose_action(env=obs[0])
                actualTime = 0.0
            else:
                actualTime += deltaTime
            
            obs = agent.step(env=obs[0],environment=env.get_environment())

    agent.save(FLAGS.filepath)

main()