import sys
import os

from tqdm import tqdm

from environment.dzwb.defeatzealots_dq import DQDefeatZealots # environment
from agents.dqagent import DQAgent # algorithm agent

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes', 40, 'Number of episodes.', lower_bound=0)
flags.DEFINE_integer('steps', 2000, 'Steps from each episode.', lower_bound=0)
flags.DEFINE_integer('episodes_for_save', 2, 'Episodes until backup save.', lower_bound=0)
flags.DEFINE_float('time_for_action', 0.2, 'Time until choose new action.', lower_bound=0.0)
flags.DEFINE_boolean('learn', False, 'Agent will learn.')
flags.DEFINE_boolean('load', False, 'Agent will load learning information. Not needed if it is not going to learn.')
flags.DEFINE_string('filepath', '\\saves\\', 'Filepath where is file for load or save.')
flags.DEFINE_string('filename', 'dzwb', 'Filename for load or save.')

FLAGS(sys.argv)

FILEPATH_SAVES = '\\saves_episode\\' + FLAGS.filename + '\\'

def main():
    #---------- INIT -----------#

    # create load and save filepath directories if they dont exist
    if not os.path.exists(os.getcwd() + FLAGS.filepath):
        os.makedirs(os.getcwd() + FLAGS.filepath)
    # create backup save filepath directories if they dont exist
    if not os.path.exists(os.getcwd() + FILEPATH_SAVES):
        os.makedirs(os.getcwd() + FILEPATH_SAVES)

    # initialize environment
    env = DQDefeatZealots()

    # initialize agent
    info = env.get_info()
    info['learn'] = FLAGS.learn
    info['episodes'] = FLAGS.episodes

    agent = DQAgent(info=info)

    if FLAGS.load or not FLAGS.learn:
        agent.load(filepath=FLAGS.filepath + FLAGS.filename)
    
    #---------- LOOP -----------#

    end = False
    ep = 0
    # environment loop
    for episode in tqdm(range(1,FLAGS.episodes+1), ascii=True, unit="episode"):
        env.reset()

        if FLAGS.learn:
            # learn reward for last action
            if end:
                agent.train()

            # backup save
            if ep >= FLAGS.episodes_for_save:
                agent.save(filepath=FILEPATH_SAVES + str(episode))
                ep = 0
            ep += 1

        env.prepare()
        agent.prepare(env=env, episode=episode-1)

        actualTime = sys.maxsize
        lastTime = 0.0

        # episode loop
        for step in range(FLAGS.steps):
            end = env.get_end()
            if end:
                break

            realTime = ((env.get_obs()).observation["game_loop"] / 16)
            deltaTime = realTime - lastTime
            lastTime = realTime

            # time to choose new action
            if actualTime >= FLAGS.time_for_action:
                agent.update(env=env, deltaTime=deltaTime)
                actualTime = 0.0
            else:
                actualTime += deltaTime
            
            env.step()
    
    agent.save(filepath=FLAGS.filepath + FLAGS.filename)

main()