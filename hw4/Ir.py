import csv
import numpy as np
import math as m


class MLE:
    def __init__(self, train_file, dictionary, eta):
        self.dictionary = read_dict(dictionary)
        self.label, self.train_file = read_file(train_file)
        self.theta = np.zeros(len(self.dictionary))
        self.b = 0
        self.eta = eta

    def train(self, num_epoch):
        for i in range(num_epoch):
            """in each loop, compute sgd of every example"""
            for index_of_train, values in self.train_file.items():
                sgd = self.compute_sgd(index_of_train)
                """update theta"""
                self.b += self.eta * sgd
                for j in values:
                    self.theta[j] += self.eta * sgd
        return self.theta, self.b

    def compute_sgd(self, exp_num):
        dot = self.dot(exp_num)
        e = m.exp(dot)
        sgd = self.label[exp_num] - e / (1 + e)
        return sgd

    def dot(self, exp_num):
        result = self.b
        for attr in self.train_file[exp_num]:
            result += self.theta[attr]
        return result

    def compute_label(self, exp_set):
        predict_label = {}
        for key in exp_set.keys():
            label = self.b
            """compute label in each exp"""
            for each_attr in exp_set[key]:
                label += self.theta[each_attr]
            predict_label[key] = label
        for key in predict_label.keys():
            if predict_label[key] < 0:
                predict_label[key] = 0
            else:
                predict_label[key] = 1
        return predict_label

    def predict(self, test_file, train_out, test_out, output_matrics):
        test_label, test_example = read_file(test_file)
        train_predict_label = self.compute_label(self.train_file)
        test_predict_label = self.compute_label(test_example)

        """print label below"""
        """train dataset"""
        # print(train_predict_label)
        file = open(train_out, "w")
        for label in sorted(train_predict_label.keys()):
            file.write(str(train_predict_label[label]) + "\n")
        file.close()
        """test dataset"""
        file = open(test_out, "w")
        for label in sorted(test_predict_label.keys()):
            file.write(str(test_predict_label[label]) + "\n")
        file.close()
        """print error rate below"""
        file = open(output_matrics, "w")
        correct_count_train = 0
        correct_count_test = 0

        for key in train_predict_label.keys():
            if train_predict_label[key] == self.label[key]:
                correct_count_train += 1

        for key in test_predict_label.keys():
            if test_predict_label[key] == test_label[key]:
                correct_count_test += 1
        file.write("error(train): {}\n".format(str(1 - correct_count_train / len(train_predict_label))))
        file.write("error(test): {}".format(str(1 - correct_count_test / len(test_predict_label))))
        file.close()


def read_dict(dictionary):
    result = {}
    with open(dictionary) as dict_file:
        reader = csv.reader(dict_file, delimiter=' ')
        for row in reader:
            result[row[0]] = row[1]
    return result


def read_file(file_name):
    examples = {}
    label = {}
    with open(file_name) as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        exp_num = 0
        for row in reader:
            label[exp_num] = int(row[0])
            x = []
            for item in row[1:-1]:
                x.append(int(item.split(":")[0]))
            examples[exp_num] = x[0:-1]
            exp_num += 1
    return label, examples


def main():
    train_file = "model1_formatted_train.tsv"
    test_file = "model1_formatted_test.tsv"
    dictionary_file = "dict.txt"
    model = MLE(train_file, dictionary_file, 0.1)
    model.train(60)
    print(model.theta, model.b)
    train_out = "train_out.labels"
    test_out = "test_out.labels"
    output_metrics = "model1_output_metrics.txt"
    """predict testing dataset"""
    model.predict(test_file, train_out, test_out, output_metrics)


if __name__ == "__main__":
    main()
