from hw8.environment import MountainCar
import numpy as np
import sys
import random as rnd


class Q_learning:
    def __init__(self, mode, episode, max_iteration, epsilon, gamma, learning_rate):
        self.mountain_car = MountainCar(mode)
        self.episode = episode
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        """"""
        self.w = np.zeros((self.mountain_car.state_space, self.mountain_car.action_space))
        self.b = 0

    def q(self, state, action):
        return np.dot(state, self.w[:, action]) + self.b

    def q_max(self, state):
        return np.max(np.dot(state, self.w) + self.b)

    def convert(self, state_dict):
        state_array = np.zeros(self.mountain_car.state_space)
        for key in state_dict.keys():
            state_array[key] = state_dict.get(key)
        return state_array

    def update(self, previous_state, action, previous_q, reward, state):
        TD_target = reward + self.gamma * self.q_max(self.convert(state))
        TD_error = previous_q - TD_target
        """"""
        self.w[:, action] -= self.learning_rate * TD_error * self.convert(previous_state)
        self.b -= self.learning_rate * TD_error

    def learn(self, return_out):
        file = open(return_out, "w")
        for episode in range(self.episode):
            state = self.mountain_car.reset()
            done = False
            total_reward = 0
            iter_count = 0
            while (not done) and iter_count < self.max_iteration:
                rand = rnd.random()
                if rand < self.epsilon:
                    action = rnd.randrange(0, 3)
                else:
                    """3 action, find the max for q"""
                    action = np.argmax(np.dot(self.convert(state), self.w))
                previous_state = state
                previous_q = self.q(self.convert(state), action)
                state, reward, done = self.mountain_car.step(action)
                self.update(previous_state, action, previous_q, reward, state)
                total_reward += reward
                iter_count += 1
            file.write("{}\n".format(total_reward))
        file.close()

    def print_w(self, weight_out):
        file = open(weight_out, "w")
        file.write("{}\n".format(self.b))
        for state in self.w:
            for action in state:
                file.write("{}\n".format(action))
        file.close()
        return


def main():
    """
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])
    """
    
    mode = "raw"
    weight_out = "weight_out"
    returns_out = "returns_out"
    episodes = 2000
    max_iterations = 200
    epsilon = 0.05
    gamma = 0.999
    learning_rate = 0.001

    q_learning = Q_learning(mode, episodes, max_iterations, epsilon, gamma, learning_rate)
    q_learning.learn(returns_out)
    q_learning.print_w(weight_out)
    pass


if __name__ == "__main__":
    main()
