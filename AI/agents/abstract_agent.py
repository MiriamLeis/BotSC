"""
    Interface
"""
class AbstractAgent:
    def prepare(self, env, episode):
        return
    def update(self, env, deltaTime):
        return
    def train(self):
        return
    def save(self, filepath):
        return
    def load(self, filepath):
        return
    def _get_max_action(self, env):
        return
    def _choose_action(self, env):
        return