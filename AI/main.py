import sys
import os

from tqdm import tqdm

from environment.bm.buildmarines_20 import BuildMarines_20States # environment
from agents.dqagent import DQAgent # algorithm agent

from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('episodes', 25, 'Number of episodes.', lower_bound=0)
flags.DEFINE_integer('steps', 14250, 'Steps from each episode.', lower_bound=0)
flags.DEFINE_integer('number_agents', 1, 'Number of agents.', lower_bound=0)
flags.DEFINE_integer('episodes_for_save', 2, 'Episodes until backup save.', lower_bound=0)
flags.DEFINE_float('time_for_action', 0.5, 'Time until choose new action.', lower_bound=0.0)
flags.DEFINE_boolean('learn', False, 'Agent will learn.')
flags.DEFINE_boolean('load', False, 'Agent will load learning information. Not needed if it is not going to learn.')
flags.DEFINE_string('filepath', '\\saves\\', 'Filepath where is file for load or save.')
flags.DEFINE_list('filename', 'bm', 'List of filename for load or save.')

FLAGS(sys.argv)

FILEPATH_SAVES = [None] * len(FLAGS.filename)
for i in range(len(FLAGS.filename)):
    FILEPATH_SAVES[i] = '\\saves_episode\\' + FLAGS.filename[i] + '\\'

def main():
    #---------- INIT -----------#

    # create load and save filepath directories if they dont exist
    if not os.path.exists(os.getcwd() + FLAGS.filepath):
        os.makedirs(os.getcwd() + FLAGS.filepath)
    # create backup save filepath directories if they dont exist
    for i in range(len(FILEPATH_SAVES)):
        if not os.path.exists(os.getcwd() + FILEPATH_SAVES[i]):
            os.makedirs(os.getcwd() + FILEPATH_SAVES[i])

    # initialize environment
    env = BuildMarines_20States()

    agents = [None] * FLAGS.number_agents
    for i in range(FLAGS.number_agents):
        info = env.get_info()
        info['learn'] = FLAGS.learn
        info['episodes'] = FLAGS.episodes
        
        # initialize agents
        agents[i] = DQAgent(info=info)
        
        env.switch()

    if FLAGS.load or not FLAGS.learn:
        i = 0
        for agent in agents:
            agent.load(filepath=FLAGS.filepath + FLAGS.filename[i])
            env.switch()
            i += 1
    
    #---------- LOOP -----------#

    end = False
    ep = 0
    # environment loop
    for episode in tqdm(range(1,FLAGS.episodes+1), ascii=True, unit="episode"):
        env.reset()

        if FLAGS.learn:
            # learn reward for last action
            if end:
                for agent in agents:
                    agent.train()
                    env.switch()

            # backup save
            if ep >= FLAGS.episodes_for_save:
                for i in range(len(agents)):    
                    agents[i].save(filepath=FILEPATH_SAVES[i] + str(episode))
                    env.switch()
                ep = 0
            ep += 1

        env.prepare()

        for agent in agents:
            agent.prepare(env=env, episode=episode-1)
            env.switch()

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
                for agent in agents:
                    agent.update(env=env, deltaTime=deltaTime)
                    env.switch()

                actualTime = 0.0
            else:
                actualTime += deltaTime
            
            env.step()
    
    for i in range(len(agents)):    
        agents[i].save(filepath=FLAGS.filepath + FLAGS.filename[i])
        env.switch()

main()