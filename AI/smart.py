import sys

from tqdm import tqdm

def smart(environment, agent, filepath, episodes, steps, time_for_action):
    agent.load(filepath)
    
    # environment loop
    for episode in tqdm(range(1,episodes+1), ascii=True, unit="episode"):
        env = environment.reset()

        agent.prepare(env=env,ep=episode-1)

        actualTime = sys.maxsize
        lastTime = 0.0

        for step in range(steps):
            # get deltaTime
            realTime = ((env).observation["game_loop"] / 16)
            deltaTime = realTime - lastTime
            lastTime = realTime

            if actualTime >= time_for_action:
                agent.update(env=env, deltaTime=deltaTime)
                agent.late_update(env=env, deltaTime=deltaTime)
                agent.get_max_action(env=env)
                actualTime = 0.0
            else:
                actualTime += deltaTime
            
            env, end = agent.step(env=env,environment=environment)

            if end:
                break
