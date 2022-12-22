import os

SAVE_PATH = 'C:\\datasets\\amazon_for_us'

count = 0
with open(os.path.join(SAVE_PATH, 'label.txt'),'r') as f:
    for line in f:
        count = count + 1
print(count)

n = open('./result/result_example_10000.txt', 'w')
for i in range(10000):
    n.writelines(str(1) + '\n')
n.close()
