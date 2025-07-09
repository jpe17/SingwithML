# 🎤 SingWithML Karaoke Application

A SingStar PlayStation-style karaoke web application that plays MIDI notes extracted from songs and overlays them with instrumental tracks.

## Features

- **SingStar-style Interface**: Beautiful, colorful interface inspired by the PlayStation SingStar game
- **Random Song Selection**: Automatically selects random songs from your extracted notes
- **Visual Note Display**: Shows notes as colorful bars positioned by pitch and time
- **Real-time Animation**: Progress bar and note highlighting during playback
- **Instrumental Overlay**: Plays backing tracks synchronized with note display
- **Responsive Design**: Works on different screen sizes

## How It Works

1. **Song Selection**: The app randomly selects a CSV file from `04_extracted_notes/`
2. **Note Visualization**: Converts MIDI notes to a visual representation with:
   - Horizontal position = time
   - Vertical position = pitch
   - Color = note name (C, D, E, etc.)
3. **Audio Playback**: Plays the corresponding instrumental track from `01_demucs/instrumental/`
4. **Synchronization**: Real-time progress bar and note highlighting

## Installation & Setup

1. **Ensure you have the virtual environment activated**:
   ```bash
   source venv/bin/activate
   ```

2. **Install Flask** (if not already installed):
   ```bash
   pip install Flask==2.3.3
   ```

3. **Run the application**:
   ```bash
   ./run_karaoke.sh
   ```
   
   Or manually:
   ```bash
   python karaoke_app.py
   ```

4. **Open your browser** and go to: `http://localhost:5000`

## Usage

1. **Load a Song**: Click "🎲 Random Song" to select a random song from your collection
2. **Start Karaoke**: Click "▶️ Play" to start the instrumental and note display
3. **Follow the Notes**: Sing along with the colored note bars as they highlight
4. **Stop Anytime**: Click "⏹️ Stop" to stop playback

## Interface Elements

- **Golden Progress Bar**: Shows current playback position
- **Colored Note Bars**: Each note is color-coded by pitch (C=red, D=blue, etc.)
- **Note States**:
  - **Upcoming**: Pulsing animation for notes coming up soon
  - **Active**: Gold border and glow for current notes
  - **Passed**: Faded appearance for completed notes
- **Status Display**: Shows current application state in bottom-right

## File Structure

```
SingwithML/
├── karaoke_app.py          # Flask web application
├── templates/
│   └── karaoke.html        # Main karaoke interface
├── run_karaoke.sh          # Startup script
├── 04_extracted_notes/     # CSV files with note data
└── 01_demucs/instrumental/ # Instrumental audio files
```

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Audio**: pygame for instrumental playback
- **Note Processing**: pandas for CSV handling, pretty_midi for MIDI operations
- **Synchronization**: Real-time WebAPI communication between frontend and backend

## Note Colors

The application uses different colors for different note names:
- **C**: Red gradient
- **C#**: Yellow-pink gradient  
- **D**: Blue gradient
- **D#**: Green gradient
- **E**: Yellow gradient
- **F**: Pink gradient
- **F#**: Purple gradient
- **G**: Orange gradient
- **G#**: Teal gradient
- **A**: Red-orange gradient
- **A#**: Blue gradient
- **B**: Blue-green gradient

## Troubleshooting

1. **No songs available**: Ensure you have CSV files in `04_extracted_notes/`
2. **No audio**: Check that instrumental files exist in `01_demucs/instrumental/`
3. **Port already in use**: Change the port in `karaoke_app.py` (line with `app.run()`)
4. **Permission errors**: Make sure `run_karaoke.sh` is executable (`chmod +x run_karaoke.sh`)

## Future Enhancements

- Microphone input for pitch detection
- Scoring system based on pitch accuracy
- Song library browser
- User profiles and high scores
- Lyrics display
- Multiple player support

Enjoy your karaoke experience! 🎵 