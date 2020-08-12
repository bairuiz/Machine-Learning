import csv
import sys
import numpy as np


class Node:
    def __init__(self, attrs_list, labels_list, attrs_name, total_data, depth, max_depth, split_attr_value=None,
                 parent=None):
        # main data
        self.attrs_list = attrs_list
        # label
        self.labels_list = labels_list
        # label name
        self.attrs_name = attrs_name
        self.total_data = total_data
        self.depth = depth
        self.max_depth = max_depth
        self.sons = []
        self.parent = parent
        # which attr select to be splited
        self.split_attr = None
        # in this case, branch of y or n
        self.split_attr_value = split_attr_value
        self.majority_vote = self.calculate_majority()

    def gini_impurity(self, data):
        label_dict = {}
        gini_impurity = 1
        for values in data:
            if values in label_dict.keys():
                label_dict[values] += 1
            else:
                label_dict[values] = 1
        for value in label_dict.values():
            gini_impurity -= (value / sum(label_dict.values())) ** 2
        return gini_impurity

    def gini_gain(self, split_data):
        gini = self.gini_impurity(self.labels_list)
        for data in split_data:
            # empty list
            if len(data) == 0:
                continue
            else:
                data1 = list(np.array(data)[:, -1])
                p = len(data) / len(self.labels_list)
                gini -= self.gini_impurity(data1) * p
        return gini

    def split_data(self, zip_data, attr_value):
        max_value = max(attr_value)
        max_list = []
        min_list = []
        result = []
        for row in zip_data:
            if row[0] == max_value:
                max_list.append(row)
            else:
                min_list.append(row)
        result.append(max_list)
        result.append(min_list)
        # print('result: ', result)
        return result

    def calculate_majority(self):
        label0 = []
        label1 = []

        max_value = max(self.labels_list)
        min_value = min(self.labels_list)
        for label in self.labels_list:
            if label == max_value:
                label0.append(label)
            else:
                label1.append(label)
        if len(label0) >= len(label1):
            return max_value
        else:
            return min_value


    def train(self):
        gini_gain_dict = {}
        '''save all the split attrs'''
        split_data = {}
        max_gini = 0
        col_num = 0

        if self.depth == self.max_depth or self.gini_impurity(self.labels_list) == 0 or len(self.total_data[0]) == 1:
            '''gini bug'''
            return self.majority_vote
        else:
            for split_attr in self.attrs_name:
                '''split_attr : label name'''
                '''attr_value: 该列的list'''
                attr_value = self.attrs_list[:, col_num]
                zip_list = list(zip(attr_value, self.labels_list))
                """4.split_data最后一个元素为[]"""
                """split_data好像有问题"""
                split_data[split_attr] = self.split_data(zip_list, attr_value)
                """3.depth = 4, 传入了一个[]"""
                gini_gain_dict[split_attr] = self.gini_gain(split_data[split_attr])
                col_num += 1

            for items in gini_gain_dict.items():
                if max_gini <= items[-1]:
                    max_gini = items[-1]
                    self.split_attr = items[0]
            """condition of shreshod"""
            if max_gini == 0:
                return self.majority_vote

            """son node below"""
            col_num = list(self.attrs_name).index(self.split_attr)
            '''y_group: rest data bo be trained'''
            '''y_group_value: vote of y in this case, branch'''
            y_group_value = max(self.attrs_list[:, col_num])
            n_group_value = min(self.attrs_list[:, col_num])
            y_group = self.total_data[0, :]
            n_group = self.total_data[0, :]
            for data in self.total_data[1:, :]:
                if data[col_num] == y_group_value:
                    y_group = np.vstack((y_group, data))
                else:
                    n_group = np.vstack((n_group, data))

            if y_group.ndim == 2:
                y_group = np.delete(y_group, col_num, axis=1)
                new_node = Node(y_group[1:, :-1], y_group[1:, -1], y_group[0, :-1],
                                y_group, self.depth + 1, self.max_depth, y_group_value, self)
                self.sons.append(new_node)
            else:
                return self.majority_vote

            if n_group.ndim == 2:
                n_group = np.delete(n_group, col_num, axis=1)
                new_node = Node(n_group[1:, :-1], n_group[1:, -1], n_group[0, :-1],
                                n_group, self.depth + 1, self.max_depth, n_group_value, self)
                self.sons.append(new_node)
            else:
                return self.majority_vote

            for son in self.sons:
                son.train()


    def print_tree(self, label0, label1):
        lable0_count = 0
        lable1_count = 0
        for label in self.labels_list:
            if label == label0:
                lable0_count += 1
            else:
                lable1_count += 1

        if self.parent is None:
            print("[{} {} /{} {}]".format(lable1_count, label1, lable0_count, label0))
        else:
            print("{}{} = {} : [{} {} /{} {}]".format("| " * self.depth,
                                                      self.parent.split_attr, self.split_attr_value, lable1_count,
                                                      label1, lable0_count, label0))
        if len(self.sons) > 0:
            for son in self.sons:
                son.print_tree(label0, label1)

    def predict(self, data, label):
        if len(self.sons) == 0:
            '''print'''
            # print(self.majority_vote, self.labels_list)
            return self.majority_vote
        else:
            col_num = list(label).index(self.split_attr)
            for son in self.sons:
                if son.split_attr_value == data[col_num]:
                    return son.predict(data, label)

    def print_predict(self, data_in):
        result = []
        label = data_in[0, :-1]
        for data in data_in[1:, :]:
            result.append(self.predict(data, label))
        return result


class DecisionTree:
    def __init__(self, file, max_depth):
        self.attr_name = None
        self.label = None
        self.attrs_list = None
        self.total_data = self.read_data(file)
        self.root = Node(self.attrs_list, self.label, self.attr_name, self.total_data, 0, max_depth)

    def read_data(self, file):
        with open(file, 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            total_data = np.array([row for row in reader])
        self.attr_name = total_data[0, :-1]
        self.label = total_data[1:, -1]
        self.attrs_list = total_data[1:, :-1]
        return total_data

    def train(self):
        self.root.train()

    def print(self):
        self.root.print_tree(max(self.label), min(self.label))

    def predict(self, test_input):
        test_data = self.read_data(test_input)
        # return self.root.print_predict(test_data[1:,:])
        return self.root.print_predict(test_data)

    def error(self, predict, in_file):
        count = 0
        in_file_label = self.read_data(in_file)[1:, -1]
        for pair in list(zip(predict, in_file_label)):
            if pair[0] != pair[1]:
                count += 1
        return count / len(predict)


if __name__ == '__main__':

    train_in = sys.argv[1]
    test_in = sys.argv[2]
    maximum_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    tree = DecisionTree(train_in, maximum_depth)
    tree.train()
    tree.print()
    predict_train = tree.predict(train_in)
    predict_test = tree.predict(test_in)
    file = open(train_out, "w")
    for output in predict_train:
        file.write("{}\n".format(output))
    file.close()

    file = open(test_out, "w")
    for output in predict_test:
        file.write("{}\n".format(output))
    file.close()

    file = open(metrics_out, "w")
    file.write("error(train): {}\n".format(tree.error(predict_train, train_in)))
    file.write("error(test): {}\n".format(tree.error(predict_test, test_in)))
    file.close()
    print(tree.error(predict_train, train_in), tree.error(predict_test, test_in))

