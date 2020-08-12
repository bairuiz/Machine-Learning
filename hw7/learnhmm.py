import sys

import numpy as np


def read_index(file_name):
    index = []
    with open(file_name, "r") as file:
        for line in file.readlines():
            index.append(line.strip("\n"))
        return index


def print_prior(prior, file_name):
    file = open(file_name, "w")
    for a in prior:
        file.write("{:.18e}\n".format(a))
    file.close()


def print_trans(trans, file_name):
    file = open(file_name, "w")
    for a in trans:
        for b in a:
            file.write("{:.18e} ".format(b))
        file.write("\n")
    file.close()


class learn:
    def __init__(self, file_name, index_to_word, index_to_tag):
        self.words = read_index(index_to_word)
        self.tags = read_index(index_to_tag)
        self.exp = self.read_file(file_name)

    def read_file(self, file_name):
        result = []
        count = 0
        with open(file_name, "r") as file:
            for line in file.readlines():
                if count >= 100:
                    break;
                lines_tmp = line.strip("\n").split(" ")
                lines = []
                count += 1
                for word in lines_tmp:
                    word_index = self.words.index(word.split("_")[0])
                    tag_index = self.tags.index(word.split("_")[1])
                    lines.append([word_index, tag_index])
                result.append(lines)
            return result

    def count_prior(self):
        pi = [1] * len(self.tags)
        for exp in self.exp:
            pi[exp[0][1]] += 1
        sum_of_pi = sum(pi)
        pi = np.array(pi)
        pi = pi / sum_of_pi
        return pi

    def count_trans(self):
        alpha = np.ones((len(self.tags), len(self.tags)))
        for exp in self.exp:
            for t in range(len(exp) - 1):
                alpha[exp[t][1], exp[t + 1][1]] += 1
        """print(alpha)"""
        for i in range(len(alpha)):
            alpha[i] = alpha[i] / sum(alpha[i])
        return alpha

    def count_emit(self):
        beta = np.ones((len(self.tags), len(self.words)))
        for exp in self.exp:
            for t in range(len(exp)):
                beta[exp[t][1], exp[t][0]] += 1
        for i in range(len(beta)):
            beta[i] = beta[i] / sum(beta[i])
        return beta


def main():

    file_name = "handout/trainwords.txt"
    index_to_word = "handout/index_to_word.txt"
    index_to_tag = "handout/index_to_tag.txt"
    hmmprior = "hmmprior.txt"
    hmmemit = "hmmemit.txt"
    hmmtrans = "hmmtrans.txt"
    """
    file_name = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    """
    learnHmm = learn(file_name, index_to_word, index_to_tag)
    print_prior(learnHmm.count_prior(), hmmprior)
    print_trans(learnHmm.count_emit(), hmmemit)
    print_trans(learnHmm.count_trans(), hmmtrans)


if __name__ == '__main__':
    main()
