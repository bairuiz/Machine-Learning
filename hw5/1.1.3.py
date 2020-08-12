import numpy as np


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
    print(alpha)
    print(x)
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
    print("NNForward: ")
    a = linear_forward(x, alpha)
    print("a: ", a.shape)
    print(a)
    z = sigmoid_forward(a)
    one = np.array([[1]])
    z_bias = np.insert(z, len(z), values=one, axis=0)
    print("z: ", z_bias)
    b = linear_forward(z_bias, beta)
    print("b: ", b)
    y_hat = softmax_forward(b)
    print("y^: ", y_hat)
    print("y:", y)
    J = cross_entropy_forward(y, y_hat)
    print("J: ", J)
    return intermediate(x, a, b, z, z_bias, y_hat, J)


def cross_entropy_backward(y, y_hat, J, g_J):
    g_y_hat = y_hat - y
    return g_y_hat


def linear_backward(z, beta, g_y_hat):
    print(z.shape, beta.shape, g_y_hat.shape)
    g_beta = np.dot(g_y_hat, z.T)
    g_z = np.dot(g_y_hat.T, beta).T
    return g_beta, g_z


def sigmoid_backward(a, z, g_z):
    """exp_a = np.divide(np.exp(-a), np.multiply(1 + np.exp(-a), 1 + np.exp(-a)))"""
    exp_a = np.multiply(z, 1-z)
    print("exp_a: ", exp_a)
    g_a = g_z * exp_a

    return g_a


def NNBackward(x, y, alpha, beta, o):
    print("NNBackward:")
    g_J = 1
    g_y_hat = cross_entropy_backward(y, o.y_hat, o.J, g_J)
    print("g_y_hat: ", g_y_hat)
    g_beta, g_z = linear_backward(o.z_bias, beta[:, :-1], g_y_hat)
    print("g_beta: ", g_beta)
    print("g_z: ", g_z)
    """g_a was wrong..."""
    g_a = sigmoid_backward(o.a, o.z, g_z)
    print("g_a: ", g_a)
    g_alpha, g_x = linear_backward(o.x, alpha, g_a)
    print("g_alpha: ", g_alpha)
    return g_alpha, g_beta


class intermediate:
    def __init__(self, x, a, b, z, z_bias, y_hat, J):
        self.x = x
        self.a = a
        self.b = b
        self.z = z
        self.z_bias = z_bias
        self.y_hat = y_hat
        self.J = J


class neural:
    def __init__(self):
        alpha = [[1, 2, -3, 0, 1, -3, 1], [3, 1, 2, 1, 0, 2, 1], [2, 2, 2, 2, 2, 1, 1], [1, 0, 2, 1, -2, 2, 1]]
        beta = [[1, 2, -2, 1, 1], [1, -1, 1, 2, 1], [3, 1, -1, 1, 1]]
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.X = np.array([[1], [1], [0], [0], [1], [1], [1]])
        self.label = np.array([[0], [1], [0]])
        print(self.X)
        print(self.alpha)
        print(self.beta)

    def train(self):
        o = NNForward(self.X, self.label, self.alpha, self.beta)
        g_alpha, g_beta = NNBackward(self.X, self.label, self.alpha, self.beta, o)
        self.beta = np.subtract(self.beta, g_beta)
        self.alpha = np.subtract(self.alpha, g_alpha)
        print()


def main():
    NN = neural()
    NN.train()
    print("updated")
    print("alpha: ", NN.alpha)
    print("beta: ", NN.beta)


if __name__ == '__main__':
    main()
