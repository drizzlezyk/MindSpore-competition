import os
from PIL import Image


ORIGIN_PATH = 'C:\\datasets\\vangogh\\data\\photo2vangogh\\train\\A'
SAVE_PATH = 'C:\\datasets\\style_transfer_final\\test'


files = os.listdir(ORIGIN_PATH)
s = []
i = 0
for file in files:
    if i == 1000:
        break
    image = Image.open(os.path.join(ORIGIN_PATH, file))
    image.save(os.path.join(SAVE_PATH, str(i)+'.jpg'))
    i = i + 1
print('finish')
