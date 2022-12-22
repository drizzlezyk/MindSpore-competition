import random
from PIL import Image
import os


def load_path(type):
    txt_path = './dataset/dataset-' + type + '.txt'
    fh = open(txt_path, 'r')
    imgs_txt = []
    for line in fh:
        line = line.rstrip()
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs_txt.append((words[0], words[1]))
    return imgs_txt


train_path_us = 'C:\\datasets\\caltech_for_us\\train\\'
train_path_user = 'C:\\datasets\\caltech_for_user\\train\\'
test_path_us = 'C:\\datasets\\caltech_for_us\\test\\'
test_path_user = 'C:\\datasets\\caltech_for_user\\test\\'

test_final_path_us = 'C:\\datasets\\caltech_for_us\\test_final\\'

train_path_list = load_path('train')
test_path_list = load_path('test')
test_final_list = load_path('test-final')
random.shuffle(test_path_list)

# save for us
for i, (path, label) in enumerate(train_path_list):
    img = Image.open(path).convert('RGB')
    save_path = train_path_us + label + '\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img.save(save_path+str(i)+'.jpg')

for i, (path, label) in enumerate(test_path_list):
    img = Image.open(path).convert('RGB')
    img.save(test_path_us+'image\\' + str(i) + '.jpg')

with open(test_path_us+'label.txt', 'w') as f:
    for i, (path, label) in enumerate(test_path_list):
        f.writelines(label+'\n')

# final
for i, (path, label) in enumerate(test_final_list):
    img = Image.open(path).convert('RGB')
    img.save(test_final_path_us+'image\\' + str(i) + '.jpg')

with open(test_final_path_us+'label-final.txt', 'w') as f:
    for i, (path, label) in enumerate(test_final_list):
        f.writelines(label+'\n')

# save for user
for i, (path, label) in enumerate(test_path_list):
    img = Image.open(path).convert('RGB')
    img.save(test_path_user+str(i)+'.jpg')

for i, (path, label) in enumerate(train_path_list):
    img = Image.open(path).convert('RGB')
    save_path = train_path_user + label + '\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img.save(save_path+str(i)+'.jpg')