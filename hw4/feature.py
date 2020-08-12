import csv


def read_file(file):
    result = []
    with open(file) as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            # print(row)
            result.append(row)
    return result


def format_data(data):
    result = []
    for row in data:
        united_data = row[0] + " " + row[1]
        result.append(united_data.strip().split(" "))
    # print(result)
    return result


def read_dict(dictionary):
    result = {}
    with open(dictionary) as dict_file:
        reader = csv.reader(dict_file, delimiter=' ')
        for row in reader:
            result[row[0]] = row[1]
    return result


def match(data, dictionary, model=1):
    result = []
    for row in data:
        index = {}
        for word in row:
            if model == 1:
                if word in dictionary:
                    index[dictionary[word]] = 1
            else:
                #index2 = {}
                if word in dictionary:
                    if dictionary[word] in index.keys():
                        index[dictionary[word]] += 1
                    else:
                        index[dictionary[word]] = 1

        if model == 2:
            for key in index.copy().keys():
                 #print(index[key])
                if index[key] < 4:
                    index[key] = 1
                else:
                    index.pop(key)

        list_of_row = [row[0], index]
        result.append(list_of_row)
    return result


def print_format_data(data, data_file):
    file = open(data_file, "w")
    for row in data:
        #print(row)
        file.write("{}\t".format(row[0]))
        for entry in row[1].items():
            file.write("{}:{}\t".format(entry[0], entry[1]))
        file.write("\n")
    file.close()


def main():
    dictionary = read_dict("dict.txt")
    # print(dictionary)
    # data = read_file("smalldata/test_data.tsv")
    # print(data)
    train_data = format_data(read_file("largedata/test_data.tsv"))
    match_data_test1 = match(train_data, dictionary)
    match_data_test2 = match(train_data, dictionary, 2)
    print_format_data(match_data_test1, "model1_formatted_test.tsv")
    print_format_data(match_data_test2, "model2_formatted_test.tsv")

    train_data = format_data(read_file("largedata/train_data.tsv"))
    match_data_test = match(train_data, dictionary)
    match_data_test2 = match(train_data, dictionary, 2)
    print_format_data(match_data_test, "model1_formatted_train.tsv")
    print_format_data(match_data_test2, "model2_formatted_train.tsv")

    train_data = format_data(read_file("largedata/valid_data.tsv"))
    match_data_test = match(train_data, dictionary)
    match_data_test2 = match(train_data, dictionary, 2)
    print_format_data(match_data_test, "model1_formatted_valid.tsv")
    print_format_data(match_data_test2, "model2_formatted_valid.tsv")


if __name__ == "__main__":
    main()
