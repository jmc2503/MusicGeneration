import time
import os
import numpy as np
import keras
import tensorflow as tf
import CustomModel
import Generator
from matplotlib import pyplot as plt

#Code derived from:
#https://www.tensorflow.org/text/tutorials/text_generation
#TensorFlow tutorial

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

#Make the last character a target so that there is a training/label data split
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

#Input processing
file = open("data3.txt", 'r')
lines = file.readlines()
input = ''.join(lines)

#Number of unique characters used in text
vocab = sorted(set(input))

#Processing to make the text into standardized ids that the network can understand easily
ids_from_chars = keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)

chars_from_ids = keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

all_ids = ids_from_chars(tf.strings.unicode_split(input, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

#Seq_length chosen for training efficiency and network output
seq_length = 100

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64

BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 512

#Create the model
model = CustomModel.myModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

checkpoint_location = os.path.join("net2_checkpoint", "ckpt_{epoch}")

#Callback allows for weights to be saved inbetween epochs and used later
callbacks_rule = [keras.callbacks.ModelCheckpoint(filepath=checkpoint_location, save_weights_only=True, monitor="loss")]

model.compile(optimizer='adam', loss=loss)

EPOCHS = 30

#Train the model
history = model.fit(dataset, epochs=EPOCHS,verbose=1,callbacks=callbacks_rule)

plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()
