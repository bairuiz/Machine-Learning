import numpy as np
import random as rnd
import sys


def readfile(filename):
    X = []
    Y = []
    with open(filename, "r") as file:
        for line in file.readlines():
            lines_tmp = line.split(",")
            entry = []
            for i in lines_tmp[1:]:
                entry.append([int(i)])
            entry.append([1])
            X.append(entry)
            y = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
            y[int(lines_tmp[0])] = [1]
            Y.append(y)
    return np.array(X), np.array(Y)


class intermediate:
    def __init__(self, x, a, b, z, z_bias, y_hat, J):
        self.x = x
        self.a = a
        self.b = b
        self.z = z
        self.z_bias = z_bias
        self.y_hat = y_hat
        self.J = J


def linear_forward(x, alpha):
    a = np.dot(alpha, x)
    return a


def sigmoid_forward(a):
    z = 1 / (1 + np.exp(-1 * a))
    return z


def softmax_forward(b):
    y_hat = np.exp(b) / np.sum(np.exp(b))
    return y_hat


def cross_entropy_forward(y, y_hat):
    y_hat = np.dot(y.T, np.log(y_hat)) * -1
    return y_hat


def NNForward(x, y, alpha, beta):
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    one = np.array([[1]])
    z_bias = np.insert(z, len(z), values=one, axis=0)
    b = linear_forward(z_bias, beta)
    y_hat = softmax_forward(b)
    J = cross_entropy_forward(y, y_hat)
    return intermediate(x, a, b, z, z_bias, y_hat, J)


def cross_entropy_backward(y, y_hat, J, g_J):
    g_y_hat = y_hat - y
    return g_y_hat


def linear_backward(z, beta, g_y_hat):
    g_beta = np.dot(g_y_hat, z.T)
    g_z = np.dot(g_y_hat.T, beta).T
    return g_beta, g_z


def sigmoid_backward(a, z, g_z):
    exp_a = np.divide(np.exp(-a), np.multiply(1 + np.exp(-a), 1 + np.exp(-a)))
    g_a = g_z * exp_a
    return g_a


def NNBackward(x, y, alpha, beta, o):
    g_J = 1
    g_y_hat = cross_entropy_backward(y, o.y_hat, o.J, g_J)
    g_beta, g_z = linear_backward(o.z_bias, beta[:, :-1], g_y_hat)
    g_a = sigmoid_backward(o.a, o.z, g_z)
    g_alpha, g_x = linear_backward(o.x, alpha, g_a)
    return g_alpha, g_beta


def predict_class(y_hat):
    list_of_y_hat = list(y_hat.T[0])
    y_class = list_of_y_hat.index(max(list_of_y_hat))
    return y_class


class neural_network:
    """D: number of examples in dataset"""

    def __init__(self, train_file, test_file, D, gamma, output_matrics, train_out, test_out, init_flag):
        self.D = D
        self.X, self.label = readfile(train_file)
        self.alpha = np.zeros((self.D, len(self.X[0])))
        self.beta = np.zeros((10, self.D + 1))
        if init_flag == 1:
            for i in range(len(self.alpha)):
                for j in range(len(self.alpha[i]) - 1):
                    self.alpha[i, j] = rnd.uniform(-1.0, 1.0)

            for i in range(len(self.beta)):
                for j in range(len(self.beta[i]) - 1):
                    self.beta[i, j] = rnd.uniform(-1.0, 1.0)

        self.gamma = gamma
        self.test_X, self.test_label = readfile(test_file)
        self.output_matrics = output_matrics
        self.train_out = train_out
        self.test_out = test_out

    def average_cross_entropy(self, X, label):
        J = 0
        for i in range(len(label)):
            single_J = NNForward(X[i], label[i], self.alpha, self.beta).J[0][0]
            J += single_J
        J /= len(X)
        return J

    def train(self, epoch=2):
        file = open(self.output_matrics, "w")
        for epoch_num in range(epoch):
            print(epoch_num)
            alpha = self.alpha
            beta = self.beta
            for i in range(len(self.X)):
                o = NNForward(self.X[i], self.label[i], alpha, beta)
                g_alpha, g_beta = NNBackward(self.X[i], self.label[i], alpha, beta, o)
                self.beta -= self.gamma * g_beta
                self.alpha -= self.gamma * g_alpha
            file.write("epoch={} crossentropy(train): {}\n".format(epoch_num + 1,
                                                                   self.average_cross_entropy(self.X, self.label)))
            file.write("epoch={} crossentropy(test): {}\n".format(epoch_num + 1,
                                                                  self.average_cross_entropy(self.test_X,
                                                                                             self.test_label)))
        file.close()

    def predict(self):
        y_hat_train = []
        y_hat_test = []
        for i in range(len(self.X)):
            y_hat_train.append(predict_class(NNForward(self.X[i], self.label[i], self.alpha, self.beta).y_hat))
        for i in range(len(self.test_X)):
            y_hat_test.append(predict_class(NNForward(self.test_X[i], self.test_label[i], self.alpha, self.beta).y_hat))
        """train_out"""
        file = open(self.train_out, "w")
        for y_hat in y_hat_train:
            file.write(str(y_hat) + "\n")
        file.close()
        """test_out"""
        file = open(self.test_out, "w")
        for y_hat in y_hat_test:
            file.write(str(y_hat) + "\n")
        file.close()
        """matrics"""
        file = open(self.output_matrics, "a")
        error_count_train = 0
        for i in range(len(self.label)):
            label = self.label[i].T[0]
            if label[y_hat_train[i]] == 0:
                error_count_train += 1
        file.write("error(train): {}\n".format(error_count_train / len(self.label)))
        error_count_test = 0
        for i in range(len(self.test_label)):
            label = self.test_label[i].T[0]
            if label[y_hat_test[i]] == 0:
                error_count_test += 1
        file.write("error(test): {}".format(error_count_test / len(self.test_label)))


def main():
    print("start...")

    train_input = "largeTrain.csv"
    test_input = "largeTest.csv"
    train_out = "model1train_out.labels"
    test_out = "model1test_out.labels"
    metrics_out = "model1metrics.txt"
    num_epoch = 100
    hidden_units = 5
    init_flag = 1
    learning_rate = 0.1
    """
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])
    """
    NN = neural_network(train_input, test_input, hidden_units, learning_rate, metrics_out, train_out, test_out,
                        init_flag)
    NN.train(num_epoch)
    NN.predict()


if __name__ == '__main__':
    main()
