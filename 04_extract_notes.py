#!/usr/bin/env python3
"""
wav_to_csv.py

Convert a WAV audio file into a CSV of note events:
start_time (s), duration (s), midi_note (0–127).

Usage:
    python wav_to_csv.py input.wav output.csv
"""
import argparse
import io
import sys
import csv

import mido
import pretty_midi
from basic_pitch.inference import predict

def wav_to_events(wav_path):
    """
    Run Basic Pitch on the input WAV and return a list of note events:
    (start_time_sec, duration_sec, midi_note)
    """
    # Run transcription
    model_output, midi_data, note_activations = predict(wav_path)

    # Convert midi_data into a mido.MidiFile
    if isinstance(midi_data, bytes):
        midi_stream = mido.MidiFile(file=io.BytesIO(midi_data))
    elif isinstance(midi_data, pretty_midi.PrettyMIDI):
        # Export PrettyMIDI object to bytes
        buf = io.BytesIO()
        midi_data.write(buf)              # writes to buffer
        buf.seek(0)
        midi_stream = mido.MidiFile(file=buf)
    else:
        raise TypeError(f"Unexpected midi_data type: {type(midi_data)}")

    # Get timing information for proper conversion to seconds
    ticks_per_beat = midi_stream.ticks_per_beat
    
    # Find tempo (default to 120 BPM if not found)
    tempo = 500000  # microseconds per beat (120 BPM)
    for track in midi_stream.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    # Convert ticks to seconds: seconds = ticks * (tempo / 1000000) / ticks_per_beat
    def ticks_to_seconds(ticks):
        return ticks * (tempo / 1000000) / ticks_per_beat

    # Iterate messages to collect note_on/off events with absolute times in seconds
    events = []
    for track in midi_stream.tracks:
        abs_time_ticks = 0
        on_times = {}
        for msg in track:
            abs_time_ticks += msg.time
            abs_time_sec = ticks_to_seconds(abs_time_ticks)
            
            if msg.type == 'note_on' and msg.velocity > 0:
                on_times[msg.note] = abs_time_sec
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                start_sec = on_times.pop(msg.note, None)
                if start_sec is not None:
                    duration_sec = abs_time_sec - start_sec
                    events.append((start_sec, duration_sec, msg.note))

    events.sort(key=lambda x: x[0])
    return events

def write_csv(events, csv_path):
    """
    Write events into a CSV file with header:
    start_time_sec,duration_sec,midi_note
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_time_sec', 'duration_sec', 'midi_note'])
        for start_time, duration, note in events:
            writer.writerow([f"{start_time:.6f}", f"{duration:.6f}", note])

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe WAV to CSV of note onsets, durations, and MIDI notes."
    )
    parser.add_argument('input_wav', help="Path to input WAV file")
    parser.add_argument('output_csv', help="Path to output CSV file")
    args = parser.parse_args()

    try:
        events = wav_to_events(args.input_wav)
        write_csv(events, args.output_csv)
        print(f"Wrote {len(events)} note events to {args.output_csv}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
