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

milestones = np.arange(0, len(os.listdir(images_path)), np.floor(len(os.listdir(images_path))/20))
for i, filename in enumerate(os.listdir(images_path)):
    if i in milestones:
        print('{:.1f}% of {} images resized...'.format(i/np.floor(len(os.listdir(images_path))/20)*5, len(os.listdir(images_path))))
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    image_arr = np.asarray(image)[...,:3] # Convert image to array and first 3 channels

training_data = np.reshape(training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('Saving...')
np.save('portrait_data.npy', training_data)
