import csv
import sys

class inspection:
    def __init__(self, train_file):
        self.label = self.read_file(train_file)
        self.label_dict = {}
        self.gini_impurity = 1
        self.error_rate = 0

    def read_file(self, train_file):
        label = []
        with open(train_file, 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            label = [row[-1] for row in reader][1:]
        return label

    def select_data(self):
        for data in self.label:
            if data in self.label_dict:
                self.label_dict[data] += 1
            else:
                self.label_dict[data] = 1

    def compute(self):
        self.error_rate = min(self.label_dict.values()) / sum(self.label_dict.values())
        for value in self.label_dict.values():
            #print(value)
            self.gini_impurity -= (value/sum(self.label_dict.values())) ** 2
        #print(self.gini_impurity)

    def print_output(self, output):
        file = open(output, "w")
        file.write('gini_impurity: {}\n'.format(round(self.gini_impurity, 4)))
        file.write('error: {}'.format(round(self.error_rate, 4)))


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    #input_file = "education_train.tsv"
    inspection = inspection(input_file)
    inspection.select_data()
    inspection.compute()
    inspection.print_output(output_file)