class AbstractAgent:
    def prepare(self, obs):
        return
    def get_info(self):
        return
    def step(self, env, func):
        return
    def update(self, obs, deltaTime):
        return
    def train(self, step, current_state, action, reward, new_state, done):
        return
    def choose_action(self, current_state):
        return
    def get_max_action(self, current_state):
        return
    def check_action_available(self, obs, action, func):
        return
    def get_reward(self, obs, action):
        return
    def get_end(self, obs):
        return