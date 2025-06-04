#!/bin/bash
# Bistoury Virtual Environment Activation Script

echo "🔧 Activating Bistoury virtual environment..."

# Check if Poetry is available and preferred
if command -v poetry &> /dev/null; then
    echo "📦 Poetry detected! Using Poetry for dependency management..."
    
    # Install dependencies with Poetry
    if ! poetry show hyperliquid-python-sdk &> /dev/null; then
        echo "📦 Installing dependencies with Poetry from pyproject.toml..."
        poetry install
        echo "✅ Poetry dependencies installed"
    fi
    
    echo "🎉 Poetry environment ready!"
    echo "📍 You are now in: $(pwd)"
    echo "🐍 Python version: $(poetry run python --version)"
    echo ""
    echo "🧪 Available test commands (Poetry):"
    echo "  poetry run pytest tests/integration/ -v    # Run integration tests"
    echo "  poetry run python tests/run_tests.py      # Run simple test suite"
    echo "  poetry shell                              # Enter Poetry shell"
    echo ""
    echo "📚 To run commands: poetry run <command>"
    
else
    echo "📦 Poetry not found. Using pip + venv..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "❌ Virtual environment not found. Creating it now..."
        python3 -m venv venv
        echo "✅ Virtual environment created"
    fi
    
    # Activate the virtual environment
    source venv/bin/activate
    
    # Check if dependencies are installed
    if ! python -c "import hyperliquid" 2>/dev/null; then
        echo "📦 Installing dependencies from pyproject.toml (editable mode)..."
        pip install --upgrade pip setuptools wheel build
        pip install -e .
        echo "📦 Installing dev dependencies (pytest, etc.)..."
        pip install pytest pytest-asyncio pytest-cov pytest-mock
        echo "✅ Dependencies installed"
    fi
    
    echo "🎉 Virtual environment activated!"
    echo "📍 You are now in: $(pwd)"
    echo "🐍 Python version: $(python --version)"
    echo ""
    echo "🧪 Available test commands (pip/venv):"
    echo "  pytest tests/integration/ -v              # Run integration tests"
    echo "  python tests/run_tests.py                 # Run simple test suite"
    echo "  PYTHONPATH=. python tests/e2e/test_hyperliquid_complete.py  # Run e2e tests"
    echo "  bistoury --help                           # Use CLI commands"
    echo ""
    echo "📚 To deactivate: deactivate"
    
fi 