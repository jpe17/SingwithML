#!/usr/bin/env python3
"""
wav_to_csv_crepe.py - With progress indicators and optimization
"""
import os
import glob
import sys
import csv
import numpy as np
import librosa
import crepe
from tqdm import tqdm
import time

# CREPE model expects 16kHz sample rate for optimal performance.
CREPE_SAMPLE_RATE = 16000

def wav_to_events(wav_path, confidence_threshold=0.8, min_note_duration=0.1):
    """Extract pitch with progress indicators"""
    
    print(f"Loading audio file: {wav_path}")
    start_time = time.time()
    
    # Load audio and resample to what CREPE expects
    audio, sr = librosa.load(wav_path, sr=CREPE_SAMPLE_RATE)
    duration = len(audio) / sr
    print(f"Audio loaded: {duration:.2f} seconds, sample rate: {sr}")
    
    # Run CREPE with only supported parameters
    print("Running CREPE pitch detection...")
    time_stamps, frequency, confidence, activation = crepe.predict(
        audio, 
        sr, 
        step_size=10,  # 10ms hop size
        verbose=1      # Show progress
    )
    
    processing_time = time.time() - start_time
    print(f"CREPE processing completed in {processing_time:.2f} seconds")
    
    # Filter by confidence
    print(f"Filtering by confidence threshold: {confidence_threshold}")
    confident_mask = confidence >= confidence_threshold
    time_stamps = time_stamps[confident_mask]
    frequency = frequency[confident_mask]
    confidence_filtered = confidence[confident_mask]
    
    print(f"Kept {len(time_stamps)}/{len(confidence)} frames ({len(time_stamps)/len(confidence)*100:.1f}%)")
    
    # Convert to MIDI notes
    print("Converting to MIDI notes...")
    midi_notes = librosa.hz_to_midi(frequency)
    midi_notes = np.round(midi_notes).astype(int)
    
    # Group into note events
    print("Grouping into note events...")
    events = group_notes(time_stamps, midi_notes, min_note_duration)
    
    print(f"Generated {len(events)} note events")
    return events

def group_notes(time_stamps, midi_notes, min_note_duration):
    """Group consecutive similar notes into events"""
    events = []
    if len(midi_notes) == 0:
        return events
    
    current_note = midi_notes[0]
    start_time = time_stamps[0]
    
    print("Grouping consecutive notes...")
    for i in range(1, len(midi_notes)):
        # Check if note changed (allow 1 semitone tolerance)
        if abs(midi_notes[i] - current_note) > 1 or (time_stamps[i] - time_stamps[i-1]) > 0.1:
            # End current note
            duration = time_stamps[i-1] - start_time
            if duration >= min_note_duration:
                events.append((start_time, duration, current_note))
            
            # Start new note
            current_note = midi_notes[i]
            start_time = time_stamps[i]
    
    # Add final note
    duration = time_stamps[-1] - start_time
    if duration >= min_note_duration:
        events.append((start_time, duration, current_note))
    
    return events

def write_csv(events, csv_path):
    """Write events to CSV file"""
    print(f"Writing {len(events)} events to {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_time_sec', 'duration_sec', 'midi_note'])
        for start_time, duration, note in events:
            writer.writerow([f"{start_time:.6f}", f"{duration:.6f}", note])

def main():
    input_dir = '03_data_preprocessing/voice'
    output_dir = '04_extracted_notes'

    # Parameters
    confidence_threshold = 0.8
    min_note_duration = 0.1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_files = glob.glob(os.path.join(input_dir, '*.wav'))

    if not wav_files:
        print(f"No .wav files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wav_files)} .wav files to process.")

    for wav_path in wav_files:
        try:
            basename = os.path.basename(wav_path)
            filename_no_ext = os.path.splitext(basename)[0]
            output_csv_path = os.path.join(output_dir, f"{filename_no_ext}.csv")

            print(f"\nProcessing {wav_path} -> {output_csv_path}")

            events = wav_to_events(wav_path, confidence_threshold, min_note_duration)

            if events:
                write_csv(events, output_csv_path)
                print(f"✓ Successfully extracted {len(events)} vocal notes to {output_csv_path}")
            else:
                print(f"No notes extracted from {wav_path} with current settings.")

        except Exception as e:
            print(f"Error processing {wav_path}: {e}", file=sys.stderr)

    print("\nAll files processed.")


if __name__ == '__main__':
    main()
