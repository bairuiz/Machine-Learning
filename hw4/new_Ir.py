import csv
import numpy as np
import math as m


def read_formated_tsv(file_path):
    X = []
    Y = []
    with open(file_path, "r") as file:
        for line in file.readlines():
            lines_tmp = line[:-1].split("\t")
            label = lines_tmp[0]
            Y.append(int(label))
            entry = {0: 1}
            for i in lines_tmp[1:]:
                k, w = i.split(":")
                entry[int(k) + 1] = 1
            X.append(entry)
    return X, Y


def read_dict(dict_path):
    dictionary = {}
    with open(dict_path) as dict_file:
        reader = csv.reader(dict_file, delimiter=' ')
        for row in reader:
            dictionary[row[0]] = row[1]
    return dictionary


def sparse_dot(theta, x):
    dot = 0.0
    for i in x.keys():
        dot += theta[i] * x[i]
    return dot


def compute_label(theta, X):
    label_dict = []
    for i in range(len(X)):
        label = sparse_dot(theta, X[i])
        label = m.exp(label) / (1 + m.exp(label))
        if label >= 0.5:
            label_dict.append(1)
        else:
            label_dict.append(0)
    return label_dict


class MLE:
    def __init__(self, train_path, dictionary, eta):
        self.X, self.Y = read_formated_tsv(train_path)
        self.dictionary = read_dict(dictionary)
        self.theta = np.zeros(len(self.dictionary) + 1)
        self.eta = eta

    def train(self, num_epoch=60):
        for epoch in range(num_epoch):
            for i in range(len(self.X)):
                dot = sparse_dot(self.theta, self.X[i])
                update = self.Y[i] - (m.exp(dot) / (1 + m.exp(dot)))
                for j in self.X[i].keys():
                    self.theta[j] += self.eta * update

    def predict(self, test_path, output_matrics, train_out, test_out):
        test_X, test_Y = read_formated_tsv(test_path)
        predict_train_label = compute_label(self.theta, self.X)
        predict_test_label = compute_label(self.theta, test_X)
        count_train_predict = 0
        count_test_predict = 0
        for i in range(len(predict_train_label)):
            if predict_train_label[i] == self.Y[i]:
                count_train_predict += 1
        for i in range(len(predict_test_label)):
            if predict_test_label[i] == test_Y[i]:
                count_test_predict += 1
        """print metrics"""
        file = open(output_matrics, "w")
        file.write("error(train): {}\n".format(str(1 - count_train_predict / len(predict_train_label))))
        file.write("error(test): {}".format(str(1 - count_test_predict / len(predict_test_label))))
        file.close()
        """print labels"""
        file = open(train_out, "w")
        for label in predict_train_label:
            file.write(str(label) + "\n")
        file.close()
        file = open(test_out, "w")
        for index in predict_test_label:
            file.write(str(predict_test_label[index]) + "\n")
        file.close()


def main():
    train_file_path = "model1_formatted_train.tsv"
    test_file_path = "model1_formatted_test.tsv"
    dict_file_path = "dict.txt"
    output_matrics = "train_output.metrics"
    train_out = "model1_train.labels"
    test_out = "model1_test.labels"
    num_epoch = 60
    eta = 0.1
    model = MLE(train_file_path, dict_file_path, eta)
    model.train(num_epoch)
    print(model.theta)
    model.predict(test_file_path, output_matrics, train_out, test_out)


if __name__ == '__main__':
    main()
