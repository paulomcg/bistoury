#!/bin/bash
# Bistoury Virtual Environment Activation Script

echo "🔧 Activating Bistoury virtual environment..."

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
    echo "📦 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
fi

echo "🎉 Virtual environment activated!"
echo "📍 You are now in: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo ""
echo "🧪 Available test commands:"
echo "  pytest tests/integration/ -v              # Run integration tests"
echo "  python tests/run_tests.py                 # Run simple test suite"
echo "  PYTHONPATH=. python tests/e2e/test_hyperliquid_complete.py  # Run e2e tests"
echo ""
echo "📚 To deactivate: deactivate" 