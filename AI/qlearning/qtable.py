import numpy as np

class QTable(object):
    def __init__(self, actions, episodes,lr=0.2, reward_decay=0.95, e_greedy=0.1,load_qt=None, load_st=None ):
        self.lr = lr
        self.actions = actions
        self.epsilon = e_greedy
        self.gamma = reward_decay
        self.load_qt = load_qt
        self.MaxEpisodes = episodes
        if load_st:
            self.states_list = self.load_states(load_st)
            set(self.states_list)
        else:
            self.states_list = set()
        
        if load_qt:
            self.q_table = self.load_qtable(load_qt)
        else:
            self.q_table = np.zeros((0, len(self.actions))) # crea la tabla Q
        
    def choose_action(self, state):
        self.check_state_exist(state)
            
        if np.random.rand() > self.epsilon:
            return np.random.choice(self.actions)
        else:
            idx = list(self.states_list).index(state)
            q_values = self.q_table[idx]
            return int(np.argmax(q_values))
    
    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        self.check_state_exist(state)

        state_idx = list(self.states_list).index(state)
        next_state_idx = list(self.states_list).index(state_)

        q_predict = self.q_table[state_idx, action]

        # almacena la recompensa por:
        # la accion que ha realizado + lo bueno que es el estado al que te ha llevado
        q_target = reward + self.gamma * self.q_table[next_state_idx].max()

        # actualizar la recompensa de ese estado con esa accion dependiendo de lo que hubiese antes
        #self.q_table[state_idx, action] = (1 - self.lr) * q_predict + self.lr * (q_target)
        self.q_table[state_idx, action] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.states_list:
            self.q_table = np.vstack([self.q_table, np.zeros((1, len(self.actions)))])
            self.states_list.add(state)
        
    def save_qtable(self, filepath):
        np.save(filepath, self.q_table)
        
    def load_qtable(self, filepath):
        return np.load(filepath)
        
    def save_states(self, filepath):
        temp = np.array(list(self.states_list))
        np.save(filepath, temp)
        
    def load_states(self, filepath):
        return np.load(filepath)

    def print_QTable(self):
        print(self.q_table)

    def set_actual_episode(self, episode):
        self.epsilon = episode/self.MaxEpisodes