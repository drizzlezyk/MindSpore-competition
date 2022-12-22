import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C


dataset_dir = "C:\\datasets\\caltech_for_user\\train"
test_dir = "C:\\datasets\\caltech_for_user\\test"

image_size = 32
mean = [0.5 * 255] * 3
std = [0.5 * 255] * 3

trans = [
    C.Resize((image_size, image_size)),
    C.Normalize(mean=mean, std=std),
    C.HWC2CHW()
]

dataset = ds.ImageFolderDataset(dataset_dir, decode=True)

dataset = dataset.map(operations=trans, num_parallel_workers=1)
