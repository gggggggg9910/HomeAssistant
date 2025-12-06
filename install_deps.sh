#!/bin/bash
# Simple dependency installation script for Ubuntu/Debian

echo "=== Home Assistant Dependency Installer ==="
echo ""

# Check if we're in a problematic environment
if [[ -f "/etc/os-release" ]] && grep -q "Ubuntu\|Debian" /etc/os-release; then
    echo "Detected Ubuntu/Debian system"

    # Check for externally-managed-environment
    if python3 -c "import sys; print('Python path:', sys.executable)" 2>/dev/null; then
        echo "Testing pip installation..."
        if ! pip install --dry-run numpy >/dev/null 2>&1; then
            echo "⚠️  Detected externally-managed-environment issue"
            echo ""

            # Try to fix with new venv
            echo "Creating fresh virtual environment..."
            rm -rf venv
            python3 -m venv venv

            if [[ -f "venv/bin/activate" ]]; then
                echo "Activating virtual environment..."
                source venv/bin/activate

                echo "Upgrading pip..."
                pip install --upgrade pip

                echo "Installing dependencies..."
                if pip install -r requirements.txt; then
                    echo ""
                    echo "✓ SUCCESS: Dependencies installed!"
                    echo ""
                    echo "To run the assistant:"
                    echo "  source venv/bin/activate"
                    echo "  python main.py"
                    exit 0
                else
                    echo ""
                    echo "✗ FAILED: Could not install dependencies"
                    echo "Trying alternative method..."
                fi
            fi

            # Alternative: try installing in user space
            echo "Trying user installation..."
            if pip install --user -r requirements.txt; then
                echo ""
                echo "✓ SUCCESS: Dependencies installed in user space!"
                echo "Note: You may need to adjust PYTHONPATH"
                exit 0
            fi

            # Last resort: break system packages (with warning)
            echo ""
            echo "⚠️  WARNING: Attempting system-wide installation (not recommended)"
            echo "This may break your system Python installation!"
            echo ""
            read -p "Continue? (y/N): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                pip install --break-system-packages -r requirements.txt
                exit $?
            fi

            echo "Installation cancelled by user"
            exit 1
        else
            echo "✓ Pip installation test passed"
            pip install -r requirements.txt
            exit $?
        fi
    else
        echo "✗ Python3 not found or not working"
        exit 1
    fi
else
    echo "Non-Ubuntu/Debian system detected, using standard installation..."
    pip install -r requirements.txt
    exit $?
fi
