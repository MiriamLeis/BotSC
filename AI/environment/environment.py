"""
    Interface
"""
class Environment:
    def reset(self):
        return
    def step(self):
        return
    def prepare(self):
        return
    def switch(self):
        return
    def update(self, deltaTime):
        return
    def get_obs(self):
        return
    def get_action(self, action):
        return
    def get_reward(self, action):
        return
    def get_end(self):
        return
    def get_info(self):
        return
    def get_state(self):
        return