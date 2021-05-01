import sys
import os

from tqdm import tqdm

from pysc2_env import PySC2 as Environment # environment
from buildmarines import BuildMarines as EnvAgent # environment agent
from dqagent import DQAgent as Agent # algorithm agent

from learn import learn
from smart import smart
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes', 500, 'Number of episodes.', lower_bound=0)
flags.DEFINE_integer('steps', 14250, 'Steps from each episode.', lower_bound=0)
flags.DEFINE_integer('episodes_for_save', 50, 'Episodes until backup save.', lower_bound=0)
flags.DEFINE_float('time_for_action', 2, 'Time until choose new action.', lower_bound=0.0)
flags.DEFINE_boolean('learn', True, 'Agent will learn.')
flags.DEFINE_boolean('load', False, 'Agent will load learning information.')
flags.DEFINE_string('filepath', 'saves\\', 'Filepath where is file for load or save.')
flags.DEFINE_string('filename', 'mtb', 'Filename for load or save.')

FLAGS(sys.argv)

FILEPATH_SAVES = 'saves_episode\\' + FLAGS.filename + '\\'

'''
def main():
    # create load and save filepath directories if they dont exist
    if not os.path.exists(FLAGS.filepath):
        os.makedirs(FLAGS.filepath)
    # create backup save filepath directories if they dont exist
    if not os.path.exists(FILEPATH_SAVES):
        os.makedirs(FILEPATH_SAVES)

    # initialize agent and environment
    agent = Agent(agent=EnvAgent(),total_episodes=FLAGS.episodes)
    if FLAGS.load:
        agent.load(FLAGS.filepath + FLAGS.filename)

    env = Environment(args=agent.get_info())

    end = False

    # environment loop
    for episode in tqdm(range(1,FLAGS.episodes+1), ascii=True, unit="episode"):
        obs = env.reset()

        if (episode % FLAGS.episodes_for_save) == 0:
            agent.save(FILEPATH_SAVES + str(episode))

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

    agent.save(FLAGS.filepath + FLAGS.filename)
'''
def main():
    # create load and save filepath directories if they dont exist
    if not os.path.exists(FLAGS.filepath):
        os.makedirs(FLAGS.filepath)
    # create backup save filepath directories if they dont exist
    if not os.path.exists(FILEPATH_SAVES):
        os.makedirs(FILEPATH_SAVES)

    # initialize agent and environment
    agent = Agent(agent=EnvAgent(),total_episodes=FLAGS.episodes)
    env = Environment(args=agent.get_info())

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