import numpy as np


class decisionStump:
    def __init__(self, train_file, split_index):
        self.train_data, self.label = self.read_file(train_file, split_index)
        self.vote_result = []
        self.error_rate_train = 0
        self.error_rate_test = 0
        self.train_predict = []
        self.test_predict = []

    def read_file(self, train_file, split_index):
        train_test_data = []
        label = []
        with open(train_file, 'r') as file:
            line = file.readline()
            line = file.readline()
            while line:
                col = line.split('\t')
                train_test_data.append(col[split_index])
                label.append(col[-1].strip('\n'))
                line = file.readline()
        #print(train_test_data)
        #print(label)
        return train_test_data, label

    def train(self):
        group0 = []
        group1 = []
        left_branch = self.train_data[0]
        for exp in list(zip(self.train_data, self.label)):
            if exp[0] == left_branch:
                group0.append(exp)
            else:
                group1.append(exp)
        # print(group0)
        # print(group1)
        if group0.count(group0[0]) >= len(group0) / 2:
            self.vote_result.append(group0[0])
        else:
            for attr in group0:
                if attr != group0[0]:
                    self.vote_result.append(attr)
                    break

        if group1.count(group1[0]) >= len(group1) / 2:
            self.vote_result.append(group1[0])
        else:
            for attr in group1:
                if attr != group1[0]:
                    self.vote_result.append(attr)
                    break
        # print(self.vote_result)

    def predict(self, test_file, split_index):
        #train_predict = []
        #test_predict = []
        """predict training dataset"""
        for data in list(zip(self.train_data, self.label)):
            if data[0] == self.vote_result[0][0]:
                self.train_predict.append(self.vote_result[0][1])
            else:
                self.train_predict.append(self.vote_result[1][1])
        # print(train_predict)
        """predict testing dataset"""
        test_data, test_label = self.read_file(test_file, split_index)
        for data in list(zip(test_data, test_label)):
            if data[0] == self.vote_result[0][0]:
                self.test_predict.append(self.vote_result[0][1])
            else:
                self.test_predict.append(self.vote_result[1][1])
        # count training error ratio
        error_train = 0
        error_test = 0
        for error_case in list(zip(self.train_predict, self.label)):
            if error_case[0] != error_case[1]:
                error_train += 1
        self.error_rate_train = error_train / len(self.label)

        for error_case in list(zip(self.test_predict, test_label)):
            if error_case[0] != error_case[1]:
                error_test += 1
        self.error_rate_test = error_test / len(test_label)
        print(self.error_rate_train)
        print(self.error_rate_test)


def main():
    train_file = 'education_train.tsv'
    test_file = 'education_test.tsv'
    split_index = 5
    stump = decisionStump(train_file, split_index)
    stump.train()
    stump.predict(test_file, split_index)


if __name__ == '__main__':
    main()