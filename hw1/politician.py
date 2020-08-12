import sys

def read_train_file(train_file, column):
    result = []
    '''politicians_train.tsv'''
    with open('C:/Users/bairu/Documents/HeinzCollege/Machine Learning/hw1/handout/' + train_file, 'r') as file:
        line = file.readline()
        line = file.readline()
        while line:
            col = line.split('\t')
            result.append((col[column - 1], col[-1].strip('\n')))
            line = file.readline()
    return result


def attr_split(attr):
    count0 = 0
    count1 = 0
    result0 = []
    result1 = []
    for exp in attr:
        if exp[0] == 'n':
            count0 += 1
            result0.append(exp)
        else:
            count1 += 1
            result1.append(exp)
    return result0, result1


def train(train_file, column):
    attr0, attr1 = attr_split(read_train_file(train_file, column))
    if attr0.count(attr0[0]) >= len(attr0) / 2:
        '''print(attr0[0])'''
        file = open("C:/Users/bairu/Documents/HeinzCollege/Machine Learning/hw1/output/testOutput.txt", "w")
        file.write(attr0[0][0] + ',' + attr0[0][1] + '\n')
        '''print('file created')'''
    else:
        i = 0
        for attr in attr0:
            if attr != attr0[0]:
                '''print(attr)'''
                file = open("C:/Users/bairu/Documents/HeinzCollege/Machine Learning/hw1/output/testOutput.txt", "w")
                file.write(attr[0] + ',' + attr[1] + '\n')
                '''print('file created')'''
                break

    if attr1.count(attr1[0]) >= len(attr1) / 2:
        '''print(attr1[0])'''
        file = open("C:/Users/bairu/Documents/HeinzCollege/Machine Learning/hw1/output/testOutput.txt", "a")
        file.write(attr1[0][0] + ',' + attr1[0][1])

    else:
        i = 0
        for attr in attr1:
            if attr != attr1[0]:
                '''print(attr)'''
                file = open("C:/Users/bairu/Documents/HeinzCollege/Machine Learning/hw1/output/testOutput.txt", "a")
                file.write(attr[0] + ',' + attr[1])
                break
    file.close()
    return 0


def read_train_output():
    with open('C:/Users/bairu/Documents/HeinzCollege/Machine Learning/hw1/output/testOutput.txt', 'r') as file:
        line = file.readline()
        result = []
        while line:
            result.append(tuple(line.strip('\n').split(',')))
            line = file.readline()
    return result


def read_test_file(test_file, column):
    result = []
    '''politicians_test.tsv'''
    with open('C:/Users/bairu/Documents/HeinzCollege/Machine Learning/hw1/handout/' + test_file, 'r') as file:
        line = file.readline()
        line = file.readline()
        while line:
            col = line.split('\t')
            result.append((col[column - 1], col[-1].strip('\n')))
            line = file.readline()
    return result


def test(train_file, test_file, column):
    stump = read_train_output()
    train_case = read_train_file(train_file, column)
    test_case = read_test_file(test_file, column)
    right_train_cnt = 0
    for case in train_case:
        if case in stump:
            right_train_cnt += 1
    train_error_ratio = 1 - right_train_cnt / len(train_case)
    right_test_cnt = 0
    for case in test_case:
        if case in stump:
            right_test_cnt += 1
    test_error_ratio = 1 - right_test_cnt / len(test_case)
    print(train_error_ratio)
    print(test_error_ratio)
    return 0


def write_ratio():
    return 0


if __name__ == "__main__":
    train_file = input("Put your train file:")
    test_file = input("Put your test file:")
    attr_col = int(input("choose attribute:"))
    train(train_file, attr_col)
    test(train_file, test_file, attr_col)
