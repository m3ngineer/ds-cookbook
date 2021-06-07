import tensorflow as tf
from tensorflow import keras

def training(dataset, epoches):
    for epoch in range(epoches):
        for batch in dataset:
            training_step(generator, discriminator, batch ,batch_size = BATCH_SIZE, k = 1)

        ## After ith epoch plot image
        if (epoch % 50) == 0:
            fake_image = tf.reshape(generator(seed), shape = (28,28))
            print("{}/{} epoches".format(epoch, epoches))
            #plt.imshow(fake_image, cmap = "gray")
            plt.imsave("{}/{}.png".format(OUTPUT_DIR,epoch),fake_image, cmap = "gray")
