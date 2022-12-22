from PIL import Image
import random

# random.seed('1eu2873870')
test_final_path_public = 'C:\\datasets\\caltech_for_us\\test_final_public\\'
test_final_path_private = 'C:\\datasets\\caltech_for_us\\test_final_private\\'


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


test_final_list = load_path('test-final')
random.shuffle(test_final_list)

length = len(test_final_list)//2

for i in range(length):
    path, label = test_final_list[i]
    img = Image.open(path).convert('RGB')
    img.save(test_final_path_public+'image\\' + str(i) + '.jpg')

with open(test_final_path_public+'label-final-public.txt', 'w') as f:
    for i in range(length):
        path, label = test_final_list[i]
        f.writelines(label+'\n')


#  ================ Private =======================

for i in range(length, length * 2):
    path, label = test_final_list[i]
    img = Image.open(path).convert('RGB')
    img.save(test_final_path_private+'image\\' + str(i) + '.jpg')

with open(test_final_path_private+'label-final-private.txt', 'w') as f:
    for i in range(length, length * 2):
        path, label = test_final_list[i]
        f.writelines(label+'\n')