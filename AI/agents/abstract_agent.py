"""
    Interface
"""
class AbstractAgent:
    def __init__(self, info):
        return
    def prepare(self, episode):
        return
    def update(self, env, deltaTime):
        return
    def save(self, filepath):
        return
    def load(self, filepath):
        return
    def _train(self):
        return
    def _get_max_action(self):
        return
    def _choose_action(self):
        return