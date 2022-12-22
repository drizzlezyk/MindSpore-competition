import mindspore.dataset as ds
import json

dataset_dir = "C:\\datasets\\image_comp\\train"

f = open("./dataset/label_dict.json", "r")

for line in f:
    label_dict = json.loads(line)
print(label_dict)

# 2) read all samples (image files) from folder cat and folder dog with label 0 and 1
imagefolder_dataset = ds.ImageFolderDataset(dataset_dir, class_indexing=label_dict)

print(imagefolder_dataset)

