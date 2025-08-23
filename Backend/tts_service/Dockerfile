# Use Python 3.11 slim image for better performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for pyttsx3 and audio processing
RUN apt-get update && apt-get install -y \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    festival \
    festvox-kallpc16k \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_minimal.txt requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create tts_outputs directory
RUN mkdir -p tts_outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port that the app runs on
EXPOSE 8007

# Command to run the application
CMD ["python", "tts.py"]