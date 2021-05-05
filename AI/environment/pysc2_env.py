from environment.environment import Environment

from pysc2.env import sc2_env
from pysc2 import maps

class PySC2(Environment):
    def __init__(self, args=[]):
        super().__init__()

        AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=64, minimap=16),
            use_feature_units=True)            

        # map_name=maps.get(class_agent.MAP_NAME)
        self.env = sc2_env.SC2Env(map_name=maps.get(args[0]),
                                players=[sc2_env.Agent(sc2_env.Race.terran)],
                                visualize=False,
                                agent_interface_format=AGENT_INTERFACE_FORMAT,
                                step_mul= 1)
                                
    def get_environment(self):
        return self.env

    def reset(self):
        super().reset()
        obs = self.env.reset()
        return obs[0]

    def step(self, actions):
        super().step(actions=actions)
        obs = self.env.step(actions=actions)
        return obs[0]

