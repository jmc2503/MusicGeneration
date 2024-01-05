import pandas as pd
import pretty_midi

#Convert note structure to midi
def notes_to_midi(notes, out_file, instrument_name):

    velocity = 100

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))

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

pitch = []
start = []
end = []
step = []
duration = []

file = open("net2_txt/net_out_5.txt", 'r')
lines = file.readlines()
for x in range(0,len(lines)):
    lines[x] = lines[x].replace("\n","")
print(lines)

for x in range(0,len(lines)):
    parts = lines[x].split()
    pitch.append(float(parts[0]))
    start.append(float(parts[1]))
    end.append(float(parts[2]))
    step.append(float(parts[3]))
    duration.append(float(parts[4]))

dict = {'pitch': pitch, 'start': start, 'end': end, 'step': step, 'duration': duration}

df = pd.DataFrame(dict)

out_file = "net2_out.midi"

out_pm = notes_to_midi(df, out_file, "Acoustic Grand Piano")
