#!/bin/bash
# Home Assistant startup script for Raspberry Pi/Ubuntu

echo "Starting Home Assistant..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "ERROR: Virtual environment not activated properly!"
    exit 1
fi

# Install/update dependencies
echo "Installing dependencies..."
if command -v pip &> /dev/null; then
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Installing packages in virtual environment..."
        pip install --upgrade pip

        # Try installing requirements
        if pip install -r requirements.txt; then
            echo "✓ Dependencies installed successfully."
        else
            echo "✗ Failed to install dependencies."
            echo ""
            echo "This might be due to Ubuntu's externally-managed-environment."
            echo "Try running the fix script:"
            echo "  ./fix_env.sh"
            echo ""
            echo "Or create a new virtual environment:"
            echo "  rm -rf venv"
            echo "  python3 -m venv venv"
            echo "  source venv/bin/activate"
            echo "  pip install -r requirements.txt"
            exit 1
        fi
    else
        echo "ERROR: Not in virtual environment!"
        echo ""
        echo "Please run one of these commands:"
        echo "1. ./start.sh (recommended)"
        echo "2. ./fix_env.sh (if you have environment issues)"
        echo "3. source venv/bin/activate && pip install -r requirements.txt"
        echo ""
        echo "If you really need to install system-wide (not recommended):"
        echo "pip install --break-system-packages -r requirements.txt"
        exit 1
    fi
else
    echo "ERROR: pip not found!"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying template..."
    cp config_template.txt .env
    echo "Please edit .env file with your API keys and configuration."
    echo ""
    echo "After editing .env, run this script again or run:"
    echo "  source venv/bin/activate && python main.py"
    exit 1
fi

# Start the assistant
echo "Starting voice assistant..."
python main.py
