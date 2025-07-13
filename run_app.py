#!/usr/bin/env python3
"""
DemucsML Application Launcher
Main entry point for the DemucsML application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main launcher function"""
    print("🎵 DemucsML: Advanced Voice-to-Instrumental Diffusion Pipeline")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("backend/app.py").exists():
        print("❌ Error: Please run this script from the DemucsML root directory")
        sys.exit(1)
    
    # Change to the backend directory
    os.chdir("backend")
    
    try:
        print("🚀 Starting Flask application...")
        print("📍 Server will be available at: http://localhost:5220")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Run the Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 