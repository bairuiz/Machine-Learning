import numpy as np
import sys


def read_index(file_name):
    index = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            index.append(line.strip("\n"))
        return index


def read_prior(file_name):
    index = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            index.append(float(line.strip(" \n")))
        return np.array(index)


def read_parameter(file_name):
    index = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            this_line = []
            for para in line.strip(" \n").split(" "):
                this_line.append(float(para))
            """index.append(line.strip("\n").split(" "))"""
            index.append(this_line)
        return np.array(index)


def log_sum_exp(alpha):
    log_alpha_T = np.log(alpha)
    if str(log_alpha_T[0]) == '-inf':
        return 2
    m = np.max(log_alpha_T)
    return m + np.log(np.sum(np.exp(log_alpha_T - m)))


class HMM:
    def __init__(self, file_name, index_to_word, index_to_tag, prior, trans, emit):
        self.words = read_index(index_to_word)
        self.tags = read_index(index_to_tag)
        self.exp = self.read_file(file_name)
        self.pi = read_prior(prior)
        self.a = read_parameter(trans)
        self.b = read_parameter(emit)

    def read_file(self, file_name):
        result = []
        count = 0
        with open(file_name, "r") as file:
            for line in file.readlines():
                lines_tmp = line.strip("\n").split(" ")
                lines = []
                for word in lines_tmp:
                    word_index = self.words.index(word.split("_")[0])
                    tag_index = self.tags.index(word.split("_")[1])
                    lines.append([word_index, tag_index])
                result.append(lines)
            return result

    def forward(self, exp):
        alpha = np.zeros((len(exp), len(self.tags)))
        for t in range(len(exp)):
            for j in range(len(self.tags)):
                if t == 0:
                    alpha[t, j] = self.pi[j] * self.b[j, exp[t][0]]
                else:
                    sum_of_a_alpha = 0
                    for k in range(len(self.tags)):
                        sum_of_a_alpha += self.a[k, j] * alpha[t - 1, k]
                    alpha[t, j] = self.b[j, exp[t][0]] * sum_of_a_alpha
        return alpha

    def backward(self, exp):
        beta = np.ones((len(exp), len(self.tags)))
        for t in range(len(exp) - 1, -1, -1):
            for j in range(len(self.tags)):
                if t == len(exp) - 1:
                    beta[t, j] = 1
                else:
                    sum_of_b_beta_alpha = 0
                    for k in range(len(self.tags)):
                        sum_of_b_beta_alpha += self.b[k, exp[t + 1][0]] * beta[t + 1, k] * self.a[j, k]
                    beta[t, j] = sum_of_b_beta_alpha
        return beta

    def predict(self, predicted, metrics):
        file = open(predicted, "w")
        sum_of_accuracy = 0
        sum_of_exp = 0
        average_log_likelihood = 0
        for exp in self.exp:
            alpha = self.forward(exp)
            beta = self.backward(exp)
            p = alpha * beta
            average_log_likelihood += log_sum_exp(alpha[len(exp) - 1])
            for i in range(len(exp)):
                file.write("{}_{}".format(self.words[exp[i][0]], self.tags[np.argmax(p[i])]))
                if i < len(exp) - 1:
                    file.write(" ")
                sum_of_exp += 1
                if np.argmax(p[i]) == exp[i][1]:
                    sum_of_accuracy += 1
            file.write("\n")
        file.close()
        sum_of_accuracy /= sum_of_exp
        average_log_likelihood /= len(self.exp)
        print(len(self.exp))
        file = open(metrics, "w")
        file.write("Average Log-Likelihood: {}\nAccuracy: {}".format(average_log_likelihood, sum_of_accuracy))


def main():

    file_name = "handout/testwords.txt"
    index_to_word = "handout/index_to_word.txt"
    index_to_tag = "handout/index_to_tag.txt"
    prior = "hmmprior.txt"
    trans = "hmmtrans.txt"
    emit = "hmmemit.txt"
    predicted = "predicted.txt"
    metrics = "metrics.txt"
    """
    file_name = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    prior = sys.argv[4]
    emit = sys.argv[5]
    trans = sys.argv[6]
    predicted = sys.argv[7]
    metrics = sys.argv[8]
    """
    hmm = HMM(file_name, index_to_word, index_to_tag, prior, trans, emit)
    hmm.predict(predicted, metrics)


if __name__ == '__main__':
    main()
