#!/bin/bash
# ResearchIQ Setup Script
# ========================
# Automated setup for ResearchIQ - Multi-Agent Research Intelligence Framework

set -e

echo "🔬 ResearchIQ Setup"
echo "===================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 10 ]); then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Upgrade pip
pip install --upgrade pip -q

# Install dependencies
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt -q

# Create .env from example
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file - please add your GEMINI_API_KEY"
fi

# Create data directories
mkdir -p data/chroma_db data/exports logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your GEMINI_API_KEY"
echo "     Get it at: https://makersuite.google.com/app/apikey"
echo ""
echo "  2. Activate the environment:"
echo "     source venv/bin/activate  (Linux/Mac)"
echo "     venv\\Scripts\\activate  (Windows)"
echo ""
echo "  3. Run the application:"
echo "     streamlit run app.py"
echo ""
echo "  4. Open: http://localhost:8501"
