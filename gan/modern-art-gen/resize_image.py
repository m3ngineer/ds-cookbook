import os
import numpy as np
from PIL import Image

# Define image size and image channel

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'dataset/'

# Define image dir path
images_path = IMAGE_DIR

training_data = []

# Iterate over images inside diretory and resize using Pillow
print('Resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    training_data.append(np.asarray(image.getdata()).reshape(128,128,3))
    print(np.asarray(image.getdata()))
    pix = np.array(image.getdata())
    # training_data.append(pix.reshape(128,128,3))


# training_data = np.reshape(training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('Saving...')
np.save('cubism_data.npy', training_data)
