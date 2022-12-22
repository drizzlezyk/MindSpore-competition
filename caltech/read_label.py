import os

datasetpath = "C:\\datasets\\caltech_competition"

d = os.listdir(datasetpath)
d.sort()
with open(r'./dataset/label.txt', 'w', encoding='utf-8') as f:
    for i in d:
        f.write(i)
        f.write('\n')


import json
label_dict = {}

with open("./dataset/label.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        line = line.split('.')
        line[0] = int(line[0])
        line[0] = str(line[0])
        label_dict[line[0]] = line[1]

print(label_dict)

with open("./dataset/label_dict.json", 'w') as outfile:
    json.dump(label_dict, outfile, ensure_ascii=False)
    # outfile.write('\n')