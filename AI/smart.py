import sys

from tqdm import tqdm

def smart(env, agent, filepath, episodes, steps, time_for_action):
    agent.load(filepath)
    
    # environment loop
    for episode in tqdm(range(1,episodes+1), ascii=True, unit="episode"):
        obs = env.reset()

        agent.prepare(env=obs[0],ep=episode-1)

        actualTime = sys.maxsize
        lastTime = 0.0

        for step in range(steps):
            # get deltaTime
            realTime = ((obs[0]).observation["game_loop"] / 16)
            deltaTime = realTime - lastTime
            lastTime = realTime

            if actualTime >= time_for_action:
                agent.update(env=obs[0], deltaTime=deltaTime)
                agent.late_update(env=obs[0], deltaTime=deltaTime)
                agent.get_max_action(env=obs[0])
                actualTime = 0.0
            else:
                actualTime += deltaTime
            
            obs, end = agent.step(env=obs[0],environment=env.get_environment())

            if end:
                break
