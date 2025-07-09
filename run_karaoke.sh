#!/bin/bash

echo "🎤 Starting SingWithML Karaoke Application..."
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run the setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install Flask if not already installed
echo "📦 Installing Flask dependencies..."
pip install Flask==2.3.3

# Start the Flask application
echo "🚀 Starting the karaoke web server..."
echo "💻 Open your browser and go to: http://localhost:5222"
echo "🎵 Press Ctrl+C to stop the server"
echo ""

python karaoke_app.py 