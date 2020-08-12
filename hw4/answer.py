#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:30:31 2019

@author: yutingwang
"""

###logistic regression !!!

import sys, os
import math


def read_csv(path):
    result = []
    labels = []
    with open(path, "r") as file:
        for line in file.readlines():
            lines_tmp = line[:-1].split("\t")
            label = lines_tmp[0]
            labels.append(int(label))
            entry = {0: 1}
            for i in lines_tmp[1:]:
                k, w = i.split(":")
                entry[int(k) + 1] = 1

            result.append(entry)

    return result, labels


# sparse vector


# x is the entry
def sparse_dot(X, W):
    product = 0.0;

    for i in X.keys():
        product += X[i] * W[i]
    return product


def SGD(X, W, label):
    tmp = math.exp(sparse_dot(X, W))
    temp = label - tmp / (1 + tmp)
    for i in X.keys():
        W[i] += 0.1 * X[i] * temp

    return W


train_path = sys.argv[1]
valid_path = sys.argv[2]
test_path = sys.argv[3]
dict_path = sys.argv[4]
train_out_path = sys.argv[5]
test_out_path = sys.argv[6]
metrics = sys.argv[7]
num_epoch = int(sys.argv[8])

# initialize vector
length = len(open(dict_path).readlines()) + 1
W = [0.] * length
result_train, labels_train = read_csv(train_path)
result_valid, labels_valid = read_csv(valid_path)
result_test, labels_test = read_csv(test_path)

for i in range(num_epoch):
    for t in range(len(result_train)):
        SGD(result_train[t], W, labels_train[t])

output_train = open(train_out_path, "w")
output_test = open(test_out_path, "w")
metrics_f = open(metrics, "w")

### train accuracy
error_train = 0.
for i in range(len(result_train)):
    val = sparse_dot(result_train[i], W)
    val = math.exp(val) / (1 + math.exp(val))
    if val >= 0.5:
        tag = 1
    else:
        tag = 0
    if (tag != labels_train[i]):
        error_train += 1
    output_train.write(str(tag) + "\n")

error_train /= len(result_train)

### test accuracy
error_test = 0.
for i in range(len(result_test)):
    val = sparse_dot(result_test[i], W)
    val = math.exp(val) / (1 + math.exp(val))
    if val >= 0.5:
        tag = 1
    else:
        tag = 0
    if (tag != labels_test[i]):
        error_test += 1
    output_test.write(str(tag) + "\n")

error_test /= len(result_test)

print("error(train): {0:.6f}".format(error_train), file=metrics_f)
print("error(test): {0:.6f}".format(error_test), file=metrics_f)
