#!/bin/bash
# Development environment activation script for pyoselm
# Usage: source scripts/activate.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME="pyoselm-dev"
VENV_PATH="$PROJECT_ROOT/.venv"

echo -e "${BLUE}🚀 PyOSELM Development Environment Setup${NC}"
echo "Project root: $PROJECT_ROOT"

# Function to create virtual environment (activation happens in main)
create_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
        python3 -m venv "$VENV_PATH"
    fi
}

# Function to upgrade pip
upgrade_pip() {
    echo -e "${YELLOW}⬆️  Upgrading pip...${NC}"
    pip install --upgrade pip
}

# Function to check if dependencies are installed
check_installation() {
    echo -e "${YELLOW}� Checking installation...${NC}"
    if python -c "import pyoselm" 2>/dev/null; then
        python -c "import pyoselm; print(f'✅ PyOSELM version: {pyoselm.__version__}')"
        echo -e "${GREEN}✅ PyOSELM is installed and ready!${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  PyOSELM not installed in development mode${NC}"
        echo -e "${YELLOW}   Run 'make install-dev' to install dependencies${NC}"
        return 1
    fi
}

# Function to show available commands
show_commands() {
    echo -e "${BLUE}📋 Next steps:${NC}"
    echo "  make install-dev   - Install development dependencies"
    echo "  make help          - Show all available make commands"
    echo "  make test          - Run tests"
    echo "  make lint          - Run linting"
    echo "  make build         - Build package"
    echo ""
    echo -e "${BLUE}📁 Scripts available:${NC}"
    echo "  scripts/publish.py - Standalone publish script"
    echo ""
    echo -e "${GREEN}🎉 Environment ready! Install dependencies with 'make install-dev'${NC}"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"

    # Check if we're already in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo -e "${YELLOW}⚠️  Already in virtual environment: $VIRTUAL_ENV${NC}"
        echo -e "${YELLOW}   Continuing with current environment...${NC}"
    else
        # Create venv if needed
        create_venv

        # Activate virtual environment (this must be done in main scope)
        echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
        source "$VENV_PATH/bin/activate"

        # Upgrade pip after activation
        upgrade_pip
    fi

    # Check if dependencies are installed (but don't install them)
    check_installation
    show_commands

    # Set useful environment variables
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export PROJECT_ROOT="$PROJECT_ROOT"

    echo -e "${GREEN}🔥 Virtual environment activated!${NC}"
}

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "${RED}❌ This script must be sourced, not executed directly.${NC}"
    echo -e "${YELLOW}Usage: source scripts/activate.sh${NC}"
    exit 1
fi

# Run main function
main
