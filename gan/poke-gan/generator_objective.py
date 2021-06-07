import tensorflow as tf
from tensorflow import keras

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def generator_objective(dx_of_gx):
    # Labels are true here because generator thinks he produces real images.
    return cross_entropy(tf.ones_like(dx_of_gx), dx_of_gx) 
