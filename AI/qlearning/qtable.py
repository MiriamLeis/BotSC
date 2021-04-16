import numpy as np
import os

class QTable(object):
    def __init__(self, actions, episodes,lr=0.2, reward_decay=0.95, e_greedy=0.9,load=False):
        self.lr = lr
        self.actions = actions
        self.epsilon = e_greedy
        self.gamma = reward_decay
        self.total_episodes = episodes

        self.states_list = set()
        self.q_table = np.zeros((0, len(self.actions))) # create Q table
    
    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        self.check_state_exist(state)

        state_idx = list(self.states_list).index(state)
        next_state_idx = list(self.states_list).index(state_)

        q_predict = self.q_table[state_idx, action]

        # almacena la recompensa por:
        # la accion que ha realizado + lo bueno que es el estado al que te ha llevado
        q_target = reward + (self.gamma * self.q_table[next_state_idx].max())

        # actualizar la recompensa de ese estado con esa accion dependiendo de lo que hubiese antes
        self.q_table[state_idx, action] = ((1 - self.lr) * q_predict) + (self.lr * (q_target))
        #self.q_table[state_idx, action] += self.lr * (q_target - q_predict)
    
    def get_max_action(self, state):
        idx = list(self.states_list).index(state)
        q_values = self.q_table[idx]
        return int(np.argmax(q_values))
        
    def choose_action(self, state):
        self.check_state_exist(state)
            
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.get_max_action(state)
        
    def save(self, filepath):
        filepath = os.getcwd() + '\\QLearning\\saves\\' + filepath

        np.save(filepath + '_qtable', self.q_table)
        temp = np.array(list(self.states_list))
        np.save(filepath + '_states', temp)
    
    def load_qtable(self, filepath):
        self.q_table = np.load(filepath)
        print("Q TABLA")
        print(type(self.q_table))
        print(self.q_table)

    def load_states(self, filepath):
        tmp_array = np.load(filepath)
        for x in tmp_array:
            self.states_list.add(x)
        print("ESTADOS")
        print(type(self.states_list))
        print(self.states_list)

    def set_epsilon(self, episode):
        self.epsilon = 1 - (episode / (self.total_episodes - (self.total_episodes / 3)))

    def __check_state_exist(self, state):
        if state not in self.states_list:
            self.q_table = np.vstack([self.q_table, np.zeros((1, len(self.actions)))])
            self.states_list.add(state)