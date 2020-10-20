from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

class PyAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(PyAgent, self).step(obs)

        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    agent = PyAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64", # Map
                    players=[sc2_env.Agent(sc2_env.Race.terran), # Our race
                             sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)], # Our enemy race and difficulty
                    agent_interface_format=features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=84, minimap=64), # Screen Dimensions
                            use_feature_units=True),
                    step_mul=8,
                    game_steps_per_episode=0,
                    visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timsteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timsteps[0])]
                    if timsteps[0].last():
                        break
                    timsteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)