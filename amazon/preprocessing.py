#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module providing dataset preprocessing"""

import json
import csv
import os.path
import random

FILE_PATH1 = "C:\\datasets\\amazon\\Books.json\\Books.json"
FILE_PATH2 = "C:\\datasets\\amazon\\Sports_and_Outdoors.json\\Sports_and_Outdoors.json"
FILE_PATH3 = "C:\\datasets\\amazon\\Video_Games.json\\Video_Games.json"
FILE_PATH4 = "C:\\datasets\\amazon\\Gift_Cards.json\\Gift_Cards.json"


def read_amazon_json(path):
    """
    :param path: path of origin json file
    :return:
    """
    data_list = []
    count = 1
    with open(path, 'r') as file:
        for line in file:
            if count > 50000:
                break
            line_json = json.loads(line)
            if line_json.get('reviewText', None) and line_json.get('overall', None):
                review = line_json['reviewText']
                label = int(float(line_json['overall']))
                if len(review) > 5 and not review.isspace():
                    data_list.append([review, label])
                    count += 1
    return data_list


data1 = read_amazon_json(FILE_PATH1)
data2 = read_amazon_json(FILE_PATH2)
data3 = read_amazon_json(FILE_PATH3)
data4 = read_amazon_json(FILE_PATH4)

all_data = data1 + data2 + data3 + data4
random.shuffle(all_data)

print(len(all_data))
BATCH_LENGTH = len(all_data)//10

SAVE_PATH = 'C:\\datasets\\amazon_for_us'

with open(os.path.join(SAVE_PATH, 'train.csv'), 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerow(["review", "label"])
    for i in range(BATCH_LENGTH * 4):
        writer.writerow(all_data[i])

with open(os.path.join(SAVE_PATH, 'test.csv'), 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerow(["review", "label"])
    for i in range(BATCH_LENGTH * 8, BATCH_LENGTH * 9):
        writer.writerow(all_data[i])

with open(os.path.join(SAVE_PATH, 'test-final.csv'), 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerow(["review", "label"])
    for i in range(BATCH_LENGTH * 9, BATCH_LENGTH * 10):
        writer.writerow(all_data[i])
mark = 0
with open(os.path.join(SAVE_PATH, 'label.txt'), 'w', newline='') as f:
    for i in range(BATCH_LENGTH * 8, BATCH_LENGTH * 9):
        f.write(str(all_data[i][1])+'\r\n')
        mark+=1
print(mark)

SAVE_PATH = 'C:\\datasets\\amazon_for_user'

with open(os.path.join(SAVE_PATH, 'train.csv'), 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerow(["review", "label"])
    for i in range(BATCH_LENGTH * 4):
        writer.writerow(all_data[i])

with open(os.path.join(SAVE_PATH, 'test.csv'), 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerow(["review", "label"])
    for i in range(BATCH_LENGTH * 8, BATCH_LENGTH * 9):
        writer.writerow([all_data[i][0], 0])

with open(os.path.join(SAVE_PATH, 'result_example.txt'), 'w', newline='') as f:
    for i in range(BATCH_LENGTH * 8, BATCH_LENGTH * 9):
        f.write(str(0)+'\r\n')




