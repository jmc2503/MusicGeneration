import os
import numpy as np
import pretty_midi
import collections
import pandas as pd
import tensorflow as tf
import keras
from matplotlib import pyplot as plt

#Code derived from:
#https://www.tensorflow.org/tutorials/audio/music_generation#create_and_train_the_model
#TensorFlow tutorial


SEQ_SIZE = 100
EPOCHS = 30

#This takes a midi file and creates a notes structure that can then be converted to usable data for the network
def create_notes(file):
    pm = pretty_midi.PrettyMIDI(file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    notes_sorted = sorted(instrument.notes, key=lambda note: note.start)
    prev = notes_sorted[0].start

    #This loop goes through and creates the necessary columns for the dataframe based on the midi file
    for x in notes_sorted:
        start = x.start
        end = x.end
        notes['pitch'].append(x.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start-prev)
        notes['duration'].append(end - start)
        prev = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

#Build the model that is actually used to train with the data
def create_model(seq_size):

    #3 inputs for the pitch, step, duration variables
    input_size= (seq_size, 3)

    #Learning rate predetermined from examples + a little bit of iterative testing
    learning_rate = 0.005

    #2 LSTM layers and 2 Dropout layers as derived from the deepjazz.io network
    inputs = keras.Input(input_size)
    x = keras.layers.LSTM(128, return_sequences=True)(inputs)
    y = keras.layers.Dropout(0.2)(x)
    z = keras.layers.LSTM(128, return_sequences=False)(y)
    a = keras.layers.Dropout(0.2)(z)

    #Output layer for each of the three inputs
    output = {
        'pitch': keras.layers.Dense(128, name='pitch')(a),
        'step': keras.layers.Dense(1, name='step')(a),
        'duration': keras.layers.Dense(1, name='duration')(a),
    }

    model = keras.Model(inputs, output)

    #Loss functions chosen so that the network can learn effectively and produce valid outputs
    loss = {
      'pitch': keras.losses.SparseCategoricalCrossentropy(
          from_logits=True),
      'step': mse_with_positive_pressure, #Allows for step and duration to only produce valid positive outputs
      'duration': mse_with_positive_pressure,
    }

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    #Scale the loss function results so that they play a more equal role
    model.compile(loss=loss,loss_weights={'pitch': 0.1, 'step': 1.0, 'duration': 1.0}, optimizer=optimizer)

    return model

#Create sequences of data so that there is training data and target data
def create_sequences(dataset, seq_length):

    vocab_size = 128
    key_order = ['pitch', 'step', 'duration']

    #This means that the original seq_length will be used to predict the next +1 characters
    seq_length = seq_length+1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1,
                                drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x/[vocab_size,1.0,1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

#For step and duration outputs
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def main():

    file_names = ["data_midi/" + file for file in os.listdir("data_midi")]
    
    #File count of the total files in the dataset
    file_count = 30
    note_list = []

    #Collect all the notes
    for file in file_names[:file_count]:
        notes = create_notes(file)
        note_list.append(notes)
    
    total_notes = pd.concat(note_list) #combine all the notes into 1 big list
    note_length = len(total_notes)

    #Create a usable dataset
    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([total_notes[key] for key in key_order], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

    seq_ds = create_sequences(notes_ds, SEQ_SIZE)

    #Batch_size chosen for realistic training time
    batch_size = 64
    buffer_size = note_length - SEQ_SIZE
    train_ds = (seq_ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE))


    #The network
    model = create_model(SEQ_SIZE)

    checkpoint_location = os.path.join("net1_checkpoint", "ckpt_{epoch}")

    #Checkpoints allow for early stopping and weights from epoch to be saved and used for predictions later
    callbacks_rule = [keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
                      keras.callbacks.ModelCheckpoint(filepath=checkpoint_location, save_weights_only=True, monitor="loss")]

    history = model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks_rule,verbose=1)

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()


main()


  



