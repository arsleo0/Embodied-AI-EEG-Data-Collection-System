#!/bin/bash
# Consciousness Research Workbench
# System Setup Script

set -e

echo "=========================================="
echo "  Consciousness Research Workbench Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
major=$(echo $python_version | cut -d'.' -f1)
minor=$(echo $python_version | cut -d'.' -f2)

if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 10 ]); then
    echo "Error: Python 3.10+ required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data experiments provenance logs

# Initialize configuration
echo ""
echo "Initializing configuration..."
if [ ! -f "config.yaml" ]; then
    cp deployment/config/production.yaml config.yaml
    echo "✓ Configuration file created"
else
    echo "✓ Configuration file already exists"
fi

# Run tests (optional)
if [ "$1" = "--test" ]; then
    echo ""
    echo "Running tests..."
    pip install pytest
    pytest tests/ -v
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start the API server:"
echo "     python scripts/start_api_server.py"
echo ""
echo "  3. Launch the dashboard:"
echo "     python scripts/launch_dashboard.py"
echo ""
echo "  4. Or use Docker:"
echo "     python scripts/deploy.py build"
echo "     python scripts/deploy.py up -d"
echo ""
