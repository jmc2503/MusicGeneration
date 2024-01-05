import keras
import tensorflow as tf
import CustomModel
import Generator
import os
import time

#Input processing
file = open("data3.txt", 'r')
lines = file.readlines()
input = ''.join(lines)

#Number of unique characters used in text
vocab = sorted(set(input))

ids_from_chars = keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)

chars_from_ids = keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

all_ids = ids_from_chars(tf.strings.unicode_split(input, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 512

model = CustomModel.myModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)

EPOCHS = 30

#Run through all the epochs like the LSTM and create predictions based on the training weights
for i in range(1,EPOCHS+1):
    checkpoint_location = os.path.join("net2_checkpoint", f"ckpt_{i}")
    model.load_weights(checkpoint_location)

    #This generates characters
    one_step_model = Generator.OneStep(model, chars_from_ids, ids_from_chars)

    start = time.time()
    states = None

    #This is the sample string I gave it to begin creating new data after
    next_char = tf.constant(['74    1.023438    1.085938  0.000000  0.062500'])
    result = [next_char]

    #This loop generates 20000 characters
    for n in range(20000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()

    #Output and decode so that it can be saved to the text file
    output = result[0].numpy().decode('utf-8')

    with open(f'net2_txt/net_out_{i}.txt', 'w') as f:
        f.write(output)

    