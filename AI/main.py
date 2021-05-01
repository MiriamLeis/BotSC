import sys
import os

from tqdm import tqdm

from environment.pysc2_env import PySC2 as Environment # environment
from agents.dzwb2v1.defeatzealots2v1_dq import DQDefeatZealots2v1 as EnvAgent # environment agent
from algorithms.dqagent import DQAgent as Agent # algorithm agent

from learn import learn
from smart import smart

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes', 1000, 'Number of episodes.', lower_bound=0)
flags.DEFINE_integer('steps', 1900, 'Steps from each episode.', lower_bound=0)
flags.DEFINE_integer('episodes_for_save', 50, 'Episodes until backup save.', lower_bound=0)
flags.DEFINE_float('time_for_action', 0.5, 'Time until choose new action.', lower_bound=0.0)
flags.DEFINE_boolean('learn', False, 'Agent will learn.')
flags.DEFINE_boolean('load', False, 'Agent will load learning information. Not needed if it is not going to learn.')
flags.DEFINE_string('filepath', '\\saves\\', 'Filepath where is file for load or save.')
flags.DEFINE_string('filename', 'dzwb1v2', 'Filename for load or save.')

FLAGS(sys.argv)

FILEPATH_SAVES = 'saves_episode\\' + FLAGS.filename + '\\'

def main():
    # create load and save filepath directories if they dont exist
    if not os.path.exists(FLAGS.filepath):
        os.makedirs(FLAGS.filepath)
    # create backup save filepath directories if they dont exist
    if not os.path.exists(FILEPATH_SAVES):
        os.makedirs(FILEPATH_SAVES)

    # initialize agent and environment
    agent = Agent(agent=EnvAgent(),total_episodes=FLAGS.episodes)
    env = Environment(args=agent.get_args())

    if FLAGS.learn:
        learn(env=env, 
            agent=agent, 
            filepath=FLAGS.filepath + FLAGS.filename, 
            saves_filepath=FILEPATH_SAVES, 
            episodes=FLAGS.episodes, 
            episodes_for_save=FLAGS.episodes_for_save, 
            steps=FLAGS.steps, 
            time_for_action=FLAGS.time_for_action, 
            load=FLAGS.load)
    else:
        smart(env=env, 
            agent=agent, 
            filepath=FLAGS.filepath + FLAGS.filename, 
            episodes=FLAGS.episodes, 
            steps=FLAGS.steps, 
            time_for_action=FLAGS.time_for_action)

main()