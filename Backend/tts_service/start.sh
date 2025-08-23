#!/bin/bash

# Startup script for TTS Service on Render
# This script ensures proper initialization of the TTS service

echo "Starting Gurukul TTS Service..."

# Set environment variables
export PYTHONPATH="/app"
export PYTHONUNBUFFERED="1"

# Create necessary directories
mkdir -p /app/tts_outputs

# Set permissions for audio directory
chmod 755 /app/tts_outputs

# Test TTS engine availability
echo "Testing TTS engine..."
python3 -c "
import pyttsx3
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print(f'TTS engine initialized successfully with {len(voices) if voices else 0} voices')
except Exception as e:
    print(f'TTS engine test failed: {e}')
    exit(1)
"

# Start the application
echo "Starting FastAPI application..."
exec python3 tts.py