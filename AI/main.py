import sys
import os

from tqdm import tqdm

from environment.pysc2_env import PySC2 as Environment # environment
from agents.bm.buildmarines import BuildMarines as EnvAgent # environment agent
from algorithms.dqagent import DQAgent as Agent # algorithm agent

from learn import learn
from smart import smart

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes', 20, 'Number of episodes.', lower_bound=0)
flags.DEFINE_integer('steps', 14250, 'Steps from each episode.', lower_bound=0)
flags.DEFINE_integer('episodes_for_save', 2, 'Episodes until backup save.', lower_bound=0)
flags.DEFINE_float('time_for_action', 0.5, 'Time until choose new action.', lower_bound=0.0)
flags.DEFINE_boolean('learn', True, 'Agent will learn.')
flags.DEFINE_boolean('load', False, 'Agent will load learning information. Not needed if it is not going to learn.')
flags.DEFINE_string('filepath', '\\saves\\', 'Filepath where is file for load or save.')
flags.DEFINE_string('filename', 'bm', 'Filename for load or save.')

FLAGS(sys.argv)

FILEPATH_SAVES = '\\saves_episode\\' + FLAGS.filename + '\\'

def main():
    # create load and save filepath directories if they dont exist
    if not os.path.exists(os.getcwd() + FLAGS.filepath):
        os.makedirs(os.getcwd() + FLAGS.filepath)
    # create backup save filepath directories if they dont exist
    if not os.path.exists(os.getcwd() + FILEPATH_SAVES):
        os.makedirs(os.getcwd() + FILEPATH_SAVES)

    # initialize agent and environment
    agent = Agent(agent=EnvAgent(),total_episodes=FLAGS.episodes)
    env = Environment(args=agent.get_args())

    if FLAGS.learn:
        learn(environment=env, 
            agent=agent, 
            filepath=FLAGS.filepath + FLAGS.filename, 
            saves_filepath=FILEPATH_SAVES, 
            episodes=FLAGS.episodes, 
            episodes_for_save=FLAGS.episodes_for_save, 
            steps=FLAGS.steps, 
            time_for_action=FLAGS.time_for_action, 
            load=FLAGS.load)
    else:
        smart(environment=env, 
            agent=agent, 
            filepath=FLAGS.filepath + FLAGS.filename, 
            episodes=FLAGS.episodes, 
            steps=FLAGS.steps, 
            time_for_action=FLAGS.time_for_action)

main()