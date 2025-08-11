#!/bin/bash

# DSPy Structured Output Explorer - One-Click Installation
# =======================================================

set -e  # Exit on any error

echo "🚀 DSPy Structured Output Explorer - One-Click Setup"
echo "=================================================="

# Install dependencies directly (no virtual env needed for demo)
echo "📥 Installing dependencies..."
pip3 install --user dspy-ai openai anthropic google-generativeai python-dotenv pydantic requests numpy

# Setup environment
echo "🔧 Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
else
    echo "⚠️  .env file already exists"
fi

# Quick test
echo "🧪 Testing installation..."
python3 -c "
try:
    import dspy, openai, pydantic
    print('✅ Installation successful!')
except ImportError as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo ""
echo "🎉 Ready to use!"
echo ""
echo "📋 Quick start:"
echo "   1. Add your API key: export OPENAI_API_KEY=sk-..."
echo "   2. Run demo: python3 quick_start.py"
echo ""
echo "📊 All demos include built-in testing - no separate test files needed!"