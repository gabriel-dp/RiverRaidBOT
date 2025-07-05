import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, file_path, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = {} 
        self.file_path = file_path
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        q_values = [self.Q.get((state_key, a), 0) for a in range(len(self.actions))]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        current_q = self.Q.get((state_key, action), 0)
        max_future_q = max([self.Q.get((next_key, a), 0) for a in range(len(self.actions))])

        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.Q[(state_key, action)] = new_q

    def save_progress(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.Q, f)

    def load_progress(self):
        try:
            with open(self.file_path, 'rb') as f:
                self.Q = pickle.load(f)
        except FileNotFoundError:
            print(f"No saved Q-table found at '{self.file_path}'. Starting fresh.")