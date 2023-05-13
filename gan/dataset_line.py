import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True, help='Path of input train dataset')
parser.add_argument('--test_path', type=str, required=True, help='Path of input test dataset')
parser.add_argument('--output_path', type=str, required=True, help='Path of output line dataset')
parser.add_argument('--output_path_test', type=str, required=True, help='Path of output line test dataset')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
opt = parser.parse_args()

if __name__ == '__main__':
    train = pd.read_csv(opt.train_path)
    test = pd.read_csv(opt.test_path)

    lines = []
    for i in range(len(train)):
        body = str(train['body'][i])
        line = body.split('.')
        lines.extend(line)

    with open(opt.output_path, 'w') as f:
        for line in lines:
            temp = line.strip()
            if len(temp) != 0:
                f.write(temp + "\n")


    lines_test = []
    for i in range(len(test)):
        body = str(test['body'][i])
        line = body.split('.')
        lines_test.extend(line)

    with open(opt.output_path_test, 'w') as f:
        for line in lines_test:
            temp = line.strip()
            if len(temp) != 0:
                f.write(temp + "\n")