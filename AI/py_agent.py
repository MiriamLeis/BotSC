import numpy as np
import pandas as pd

from pysc2.env import sc2_env
from absl import app

step_mul = 8

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(state)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.ix[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()

        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        self.check_state_exist(state)

        q_predict = self.q_table.ix[state, action]
        q_target = reward + self.gamma * self.q_table.ix[state_, :].max()

        self.q_table.ix[state, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.colums, name=state))

def main(unused_argv):
    with sc2_env.SC2Env(
            map_name="MoveToBeacon",
            step_mul=step_mul,
            visualize=True) as env:




if __name__ == "__main__":
    app.run(main)