#!/bin/bash
# Fix virtual environment issues

echo "=== Environment Diagnostics ==="
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "PATH: $PATH"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo "Python version: $(python --version 2>&1)"
echo "Pip version: $(pip --version 2>&1)"
echo ""

# Check if we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo "⚠️  WARNING: You are in a conda environment: $CONDA_DEFAULT_ENV"
    echo "Conda environments may have this issue. Try using venv instead."
    echo ""
fi

# Check if pip is from virtual environment
if [[ "$VIRTUAL_ENV" != "" ]] && [[ "$(which pip)" == *"$VIRTUAL_ENV"* ]]; then
    echo "✓ Pip is from virtual environment"
else
    echo "✗ Pip is NOT from virtual environment"
    echo "Expected pip path to contain: $VIRTUAL_ENV"
    echo "Actual pip path: $(which pip)"
    echo ""
fi

echo "=== Attempting to fix ==="

# Method 1: Create a new venv in project directory
if [[ ! -d "venv" ]]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
fi

echo "Activating new virtual environment..."
source venv/bin/activate

echo "New environment status:"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Pip path: $(which pip)"

# Install packages
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

if [[ $? -eq 0 ]]; then
    echo "✓ Installation successful!"
else
    echo "✗ Installation failed"
    echo ""
    echo "Trying with --break-system-packages (not recommended)..."
    pip install --break-system-packages -r requirements.txt
fi

echo ""
echo "To use this environment in the future:"
echo "source venv/bin/activate"
echo "python main.py"
