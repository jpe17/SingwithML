import os
import json
import time
import random
import sys
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

from model_integration import get_model_integration

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = '../uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('../frontend/static/audio', exist_ok=True)

# Initialize model integration
app.config['MODEL_INTEGRATION'] = get_model_integration()

# Sample data for demo examples (updated paths)
DEMO_EXAMPLES = [
    {
        'id': 'smashmouth',
        'title': 'All Star',
        'artist': 'Smash Mouth',
        'original_file': '../frontend/static/audio/originals/041000_smashmouthallstarlyrics.mp3',
        'vocal_file': '../demucs/separated/01_demucs/voice/voice_041000_smashmouthallstarlyrics.wav',
        'reference_instrumental': '../demucs/separated/01_demucs/instrumental/instrumental_041000_smashmouthallstarlyrics.wav',
        'genre': 'Pop Rock',
        'year': '1999',
        'duration': '3:21'
    },
    {
        'id': 'queen',
        'title': 'Bohemian Rhapsody', 
        'artist': 'Queen',
        'original_file': '../frontend/static/audio/originals/037000_queenbohemianrhapsodyofficialvideoremastered.mp3',
        'vocal_file': '../demucs/separated/01_demucs/voice/voice_037000_queenbohemianrhapsodyofficialvideoremastered.wav',
        'reference_instrumental': '../demucs/separated/01_demucs/instrumental/instrumental_037000_queenbohemianrhapsodyofficialvideoremastered.wav',
        'genre': 'Rock',
        'year': '1975',
        'duration': '5:55'
    },
    {
        'id': 'journey',
        'title': "Don't Stop Believin'",
        'artist': 'Journey',
        'original_file': '../frontend/static/audio/originals/028000_journeydontstopbelievinofficialaudio.mp3',
        'vocal_file': '../demucs/separated/01_demucs/voice/voice_028000_journeydontstopbelievinofficialaudio.wav',
        'reference_instrumental': '../demucs/separated/01_demucs/instrumental/instrumental_028000_journeydontstopbelievinofficialaudio.wav',
        'genre': 'Rock',
        'year': '1981',
        'duration': '4:11'
    }
]

@app.route('/')
def index():
    return render_template('index.html', examples=DEMO_EXAMPLES)

@app.route('/api/demo/<demo_id>')
def get_demo(demo_id):
    """Get demo data for a specific example"""
    demo = next((d for d in DEMO_EXAMPLES if d['id'] == demo_id), None)
    if not demo:
        return jsonify({'error': 'Demo not found'}), 404
    
    return jsonify(demo)

@app.route('/api/generate', methods=['POST'])
def generate_instrumental():
    """Use the real diffusion model to generate instrumental"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"generated_{unique_filename}")
        file.save(input_path)
        
        # Progress tracking
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        # Use real model integration
        model_int = app.config['MODEL_INTEGRATION']
        result = model_int.generate_instrumental(
            input_path, 
            output_path, 
            progress_callback
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'file_id': unique_filename,
                'output_file': f"generated_{unique_filename}",
                'progress_updates': progress_updates,
                'message': result['message'],
                'model_used': result.get('model_used', 'unknown'),
                'estimated_time': 30
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'message': result['message']
            }), 500

@app.route('/api/generate_demo', methods=['POST'])
def generate_demo():
    """Generate instrumental for demo examples using real model"""
    demo_vocal_path = request.form.get('demo_vocal_path')
    if not demo_vocal_path:
        return jsonify({'error': 'No demo vocal path provided'}), 400
    
    # Create output filename
    timestamp = str(int(time.time()))
    output_filename = f"demo_{timestamp}.wav"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # Progress tracking
    progress_updates = []
    
    def progress_callback(update):
        progress_updates.append(update)
    
    # Use real model integration
    model_int = app.config['MODEL_INTEGRATION']
    result = model_int.generate_instrumental(
        demo_vocal_path, 
        output_path, 
        progress_callback
    )
    
    if result['success']:
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'progress_updates': progress_updates,
            'message': 'Demo instrumental generated using real model',
            'model_used': result.get('model_used', 'real')
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error'],
            'message': result['message']
        }), 500

@app.route('/api/progress/<file_id>')
def get_progress(file_id):
    """Simulate real-time progress updates"""
    # In a real implementation, you'd track actual progress
    progress = random.randint(40, 95)
    current_step = random.choice([
        "Denoising step 23/50...",
        "Applying vocal conditioning...", 
        "Refining instrumental features...",
        "Generating harmonic content..."
    ])
    
    return jsonify({
        'progress': progress,
        'current_step': current_step,
        'eta': random.randint(5, 15)
    })

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files"""
    return send_file(filename)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5220) 