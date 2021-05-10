import sys

from tqdm import tqdm

def learn(environment, agent, filepath, saves_filepath, episodes, episodes_for_save, steps, time_for_action, load):
    if load:
        agent.load(filepath)
    
    end = False
    ep = 0
    # environment loop
    for episode in tqdm(range(1,episodes+1), ascii=True, unit="episode"):
        env = environment.reset()

        # learn reward for last action
        if end:
            agent.train()

        # backup save
        if ep == episodes_for_save:
            agent.save(saves_filepath + str(episode))
            ep = 0
        ep += 1

        agent.prepare(env=env,ep=episode-1)

        actualTime = sys.maxsize
        lastTime = 0.0

        for step in range(steps):
            if agent.get_end(env):
                break

            realTime = ((env).observation["game_loop"] / 16)
            deltaTime = realTime - lastTime
            lastTime = realTime

            if actualTime >= time_for_action:
                agent.update(env=env, deltaTime=deltaTime)
                agent.train()
                agent.late_update(env=env, deltaTime=deltaTime)
                agent.choose_action(env=env)
                actualTime = 0.0
            else:
                actualTime += deltaTime
            
            env = agent.step(env=env,environment=environment)
    
    agent.save(filepath)
