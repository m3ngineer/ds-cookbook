from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# Preview image frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100

# Size vector to generate images from
NOISE_SIZE = 100

# Configuration
EPOCHS = 1200 # Number of iterations
BATCH_SIZE = 32 # Number of images to feed in each iteration

GENERATE_RES = 3
IMAGE_SIZE = 128 # rows/cols

IMAGE_CHANNELS = 3

# Load data
training_data = np.load('cubism_data.py')

# Create discriminator function
def build_discriminator(image_shape):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2))
    input_shape = image_shape, padding="same"))
    model.add(LeakyRelU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=(0.1, (0.1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding=”same”))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    mode.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    input_image = Input(shape=image_shape)
    validity = model(input_image)

    return Model(input_image, validity)

# Create generator function
def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation=”relu”,       input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding=”same”))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation(“relu”))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding=”same”))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation(“relu”))
    for i in range(GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding=”same”))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation(“relu”))
    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding=”same”))
    model.add(Activation(“tanh”))
    input = Input(shape=(noise_size,))
    generated_image = model(input)

    return Model(input, generated_image)

def save_images(cnt, noise):
    ''' Generates frames from cnt and noise params and saves image array '''
    
    image_array = np.full((PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
                    PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),
                    255, dtype=np.unit8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c +
                        IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1

    output_path = 'output'
    if not os.path.exists(output_path):
        os.makesirs(output_path)

    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
