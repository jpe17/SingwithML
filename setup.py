#!/usr/bin/env python3
"""
DemucsML Setup Script
Handles initial setup and dependency installation
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def setup_directories():
    """Create necessary directories"""
    directories = [
        "uploads",
        "backend/models/checkpoints",
        "demucs/separated/01_demucs/voice",
        "demucs/separated/01_demucs/instrumental",
        "frontend/static/audio/originals",
        "00_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    """Main setup function"""
    print("🎵 DemucsML Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Install DEMUCS separately (it's a large package)
    print("\n🎼 Installing DEMUCS...")
    if not run_command("pip install demucs", "Installing DEMUCS"):
        print("⚠️  DEMUCS installation failed. You may need to install it manually.")
    
    # Check for system dependencies
    print("\n🔍 Checking system dependencies...")
    
    # Check for ffmpeg
    if run_command("ffmpeg -version", "Checking FFmpeg"):
        print("✅ FFmpeg is available")
    else:
        print("⚠️  FFmpeg not found. Please install it for audio processing:")
        print("   - macOS: brew install ffmpeg")
        print("   - Ubuntu: sudo apt install ffmpeg")
        print("   - Windows: Download from https://ffmpeg.org/")
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Place your MP3 files in the '00_data' directory")
    print("2. Run source separation: python demucs/scripts/extract_voice.py")
    print("3. Start the web application: python run_app.py")
    print("\n📖 For more information, see the README.md file")

if __name__ == "__main__":
    main() 