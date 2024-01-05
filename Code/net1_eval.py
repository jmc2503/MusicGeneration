import tensorflow as tf
import keras
import os
import numpy as np
import pretty_midi
import pandas as pd
import collections
from matplotlib import pyplot as plt

SEQ_SIZE = 100
EPOCHS=30

#Code derived from:
#https://www.tensorflow.org/tutorials/audio/music_generation#create_and_train_the_model
#TensorFlow tutorial

#Creates the neural network model used to train the data upon
def create_model(seq_size):
    #3 inputs represent the pitch, step, duration
    input_size= (seq_size, 3)

    #Learning rate decided from given value and slight deviations which led to not much change
    learning_rate = 0.005

    #2 LSTM layers and 2 Dropout Layers derived from deepjazz.io network
    inputs = keras.Input(input_size)
    x = keras.layers.LSTM(128, return_sequences=True)(inputs)
    y = keras.layers.Dropout(0.2)(x)
    z = keras.layers.LSTM(128, return_sequences=False)(y)
    a = keras.layers.Dropout(0.2)(z)

    #Output layer gives 3 outputs
    output = {
        'pitch': keras.layers.Dense(128, name='pitch')(a),
        'step': keras.layers.Dense(1, name='step')(a),
        'duration': keras.layers.Dense(1, name='duration')(a),
    }

    model = keras.Model(inputs, output)

    #Loss functions defined for each output based on what the output cares about most
    loss = {
      'pitch': keras.losses.SparseCategoricalCrossentropy(
          from_logits=True),
      'step': mse_with_positive_pressure, #positive pressure keeps step and duration positive for usable outputs
      'duration': mse_with_positive_pressure,
    }

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss,loss_weights={'pitch': 0.1, 'step': 1.0, 'duration': 1.0}, optimizer=optimizer)

    return model

#Takes in notes and a model and uses that to predict what the next note to be, temperature adds randomness
def predict_next_note(notes, model, temperature):

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    #Make a prediction based on the trained network
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    #Temperature causes scaling of the probability so that the same choice is not made every time 
    #This is a problem that I saw many music neural networks fall into in examples so temperature helps to mitigate that
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # step and duration values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


#For step and duration
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)

#This takes a midi file and creates a notes structure that can then be converted to usable data for the network
def create_notes(file):
    pm = pretty_midi.PrettyMIDI(file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    notes_sorted = sorted(instrument.notes, key=lambda note: note.start)
    prev = notes_sorted[0].start

    #Creates effectively columns for each of the required data points
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

#Converts a notes structure to a midi file at out_file location
def notes_to_midi(notes, out_file, instrument_name):

    velocity = 100

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

    #Iterate through all the notes and perform calculations to place the notes where they should be in the midi
    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

def main():

    model = create_model(SEQ_SIZE)

    note_data = []

    key_order = ['pitch', 'step', 'duration']
    file_names = ["data_midi/" + file for file in os.listdir("data_midi")]

    #This loops goes through all the checkpoints from the training epochs and produces a midi file
    #This midi file allows us to see how the network changed as it was trained
    for i in range(1,EPOCHS+1):
        checkpoint_location = os.path.join("net1_checkpoint", f"ckpt_{i}")
        model.load_weights(checkpoint_location)

        #Predetermined values
        temperature = 2
        num_predictions = 120

        #Test with a MIDI file that the network was trained on to give it a chance of producing something good
        test_notes = create_notes(file_names[10])
        test_midi = pretty_midi.PrettyMIDI(file_names[10])
        instrument = test_midi.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

        sample_notes = np.stack([test_notes[key] for key in key_order], axis=1)


        input_notes = (sample_notes[:SEQ_SIZE] / np.array([128, 1, 1]))

        #Generate the notes and save them into a dataframe that can then be converted to a midi
        generated_notes = []
        prev_start = 0
        for _ in range(num_predictions):
            pitch, step, duration = predict_next_note(input_notes, model, temperature)
            start = prev_start + step
            end = start + duration
            input_note = (pitch, step, duration)
            generated_notes.append((*input_note, start, end))
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))

        num_notes = 0

        #Count the number of notes that are actually played, this was a bigger problem in initial tests
        #but I kept it just in case
        for note in generated_notes['duration']:
            if note > 0:
                num_notes += 1
        
        note_data.append(num_notes)

        out_file = f"net1_midi/output_{i}.midi"
        out_pm = notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name)


    plt.plot(range(1,EPOCHS+1),note_data, label='notes per epoch')
    plt.show()

main()