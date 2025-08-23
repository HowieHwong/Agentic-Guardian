#!/bin/bash

# AuraGen Documentation Build Script
# This script sets up and builds the Sphinx documentation

set -e  # Exit on any error

echo "🚀 Building AuraGen Documentation"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "source/conf.py" ]; then
    echo "❌ Error: Please run this script from the docs/ directory"
    exit 1
fi

# Install documentation dependencies
echo "📦 Installing documentation dependencies..."
pip install -r requirements.txt

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/

# Build HTML documentation
echo "🔨 Building HTML documentation..."
sphinx-build -b html source build/html

# Check for warnings
if [ $? -eq 0 ]; then
    echo "✅ Documentation built successfully!"
    echo ""
    echo "📖 You can view the documentation by opening:"
    echo "   file://$(pwd)/build/html/index.html"
    echo ""
    echo "🌐 Or serve it locally with:"
    echo "   cd build/html && python -m http.server 8000"
    echo "   Then visit: http://localhost:8000"
else
    echo "❌ Documentation build failed!"
    exit 1
fi

# Optional: Open in browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "🔗 Open documentation in browser? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "build/html/index.html"
    fi
fi

echo "🎉 Documentation build complete!"
