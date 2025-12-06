#!/bin/bash
# Home Assistant startup script for Raspberry Pi

echo "Starting Home Assistant..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying template..."
    cp config_template.txt .env
    echo "Please edit .env file with your API keys and configuration."
    exit 1
fi

# Start the assistant
echo "Starting voice assistant..."
python main.py
