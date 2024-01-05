import pretty_midi
import os
import pandas as pd
import collections
import numpy as np

file_names = ["data_midi/" + file for file in os.listdir("data_midi")]
file_count = 10

lines = []

def midi_to_notes(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

for file in file_names[:file_count]:
    frame = midi_to_notes(file)
    string_frame = frame.to_string(header=False)
    splitter = string_frame.split("\n")
    for string in splitter:
        lines.append(string[6:])

with open('data3.txt', 'w') as f:
    for line in lines:
        f.write(line)
        f.write("\n")
