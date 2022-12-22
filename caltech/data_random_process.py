import os
import random


dataset_path = "C:\\datasets\\caltech_competition"

dirs = os.listdir(dataset_path)
dirs.sort()

NUM_CLASS = 256

it = 0
Matrix = [[] for x in range(NUM_CLASS)]
for d in dirs:
    for _, _, filename in os.walk(os.path.join(dataset_path, d)):
        for i in filename:
            Matrix[it].append(os.path.join(os.path.join(dataset_path, d), i))
    it = it + 1


random_matrix = [[] for x in range(NUM_CLASS)]
start_trn_idx = [0] * NUM_CLASS

with open(r'./dataset/dataset-test.txt', 'w', encoding='utf-8') as f:
    for i, _ in enumerate(Matrix):
        random_matrix[i] = random.sample(list(range(0, len(Matrix[i]))), 20)
        for j in random_matrix[i]:
            f.write(os.path.join(dataset_path, Matrix[i][j]))
            f.write(' ')
            f.write(str(i+1))
            f.write('\n')


with open(r'dataset/dataset-test-final.txt', 'w', encoding='utf-8') as f:
    for i, _ in enumerate(Matrix):
        count = 0
        for j in range(len(Matrix[i])):
            if j not in random_matrix[i]:
                f.write(os.path.join(dataset_path, Matrix[i][j]))
                f.write(' ')
                f.write(str(i+1))
                f.write('\n')
                count = count + 1
            if count == 20:
                start_trn_idx[i] = j + 1
                break


with open(r'./dataset/dataset-train.txt', 'w', encoding='utf-8') as f:
    for i, _ in enumerate(Matrix):
        start = start_trn_idx[i]
        for j in range(start, len(Matrix[i])):
            if j not in random_matrix[i]:
                f.write(os.path.join(dataset_path, Matrix[i][j]))
                f.write(' ')
                f.write(str(i+1))
                f.write('\n')
